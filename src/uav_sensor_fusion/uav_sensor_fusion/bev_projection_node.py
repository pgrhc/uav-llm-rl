"""
BEV Projection Node v2 - Düzeltilmiş Versiyon (+ YOLO Projected Detections Publisher)

Eklenenler:
- YOLO bbox -> ray-plane -> ground point (odom) hesaplandıktan sonra
  bu nokta base_link'e dönüştürülüp /yolo/projected_detections topic'ine yayınlanır.
- Mesaj tipi: vision_msgs/Detection3DArray
  Her Detection3D içinde:
    - results[0].hypothesis.class_id  (YOLO class_id)
    - results[0].hypothesis.score     (YOLO score)
    - results[0].pose.pose.position   (base_link'te x_rel, y_rel, z=0)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from vision_msgs.msg import Detection2DArray

# ✅ NEW imports for projected detections
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose

from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
from fusion_msgs.msg import RadarPoints

import numpy as np
import cv2
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs


class BEVImageNodeV2(Node):
    def __init__(self):
        super().__init__("bev_image_node_v2")

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # BEV Grid Parametreleri
        self.grid_size = 600          # 600x600 piksel
        self.res = 0.05               # 5 cm/piksel -> 30m x 30m alan

        # Kamera Intrinsics (Gazebo kamera modeli)
        self.fx = 539.9
        self.fy = 539.9
        self.cx = 640.0
        self.cy = 480.0
        self.camera_info_received = False

        # CameraInfo
        self.create_subscription(
            CameraInfo,
            "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/camera_info",
            self.camera_info_callback,
            10
        )

        # QoS for PX4 (şu an kullanılmıyor ama bırakıyorum)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pub_lidar = self.create_publisher(Image, "/bev/lidar_layer", 10)
        self.pub_radar = self.create_publisher(Image, "/bev/radar_layer", 10)
        self.pub_yolo_layer = self.create_publisher(Image, "/bev/yolo_layer", 10)

        # Subscribers
        self.create_subscription(
            PointCloud2,
            "/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan/points",
            self.lidar_callback,
            10
        )
        self.create_subscription(
            Detection2DArray,
            "/yolo/detections",
            self.yolo_callback,
            10
        )
        self.create_subscription(
            RadarPoints,
            "/radar/points_filtered_radarmsg",
            self.radar_callback,
            10
        )

        # Publishers
        self.pub = self.create_publisher(Image, "/bev/image", 10)

        # ✅ NEW publisher: YOLO projected detections in base_link
        self.pub_yolo_proj = self.create_publisher(
            Detection3DArray,
            "/yolo/projected_detections",
            10
        )

        # Latest data storage
        self.latest_lidar = None
        self.latest_yolo = None
        self.latest_radar = None

        # Drone pozisyonu (drone-relative BEV için)
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0

                # --- Timing / sync ---
        self.max_age_sec = 0.5   # 0.3-0.8 arası deneyebilirsin
        self.create_timer(0.1, self.process)  # 10 Hz sabit BEV üretimi

        self.get_logger().info("BEV Node V2 Started - Drone-Relative BEV (+ projected detections)")

    # ==================== CALLBACKS ====================
    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.camera_info_received = True

    def lidar_callback(self, msg):
        self.latest_lidar = msg

    def yolo_callback(self, msg):
        self.latest_yolo = msg
        # self.process()

    def radar_callback(self, msg):
        self.latest_radar = msg

    def _stamp_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    # ==================== MAIN PROCESS ====================
    def process(self):
        if not self.camera_info_received:
            self.get_logger().warn("Waiting for Camera Info...", throttle_duration_sec=2)
            return
        if self.latest_lidar is None or self.latest_yolo is None or self.latest_radar is None:
            return
        
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        t_lidar = self._stamp_to_sec(self.latest_lidar.header.stamp)
        t_yolo  = self._stamp_to_sec(self.latest_yolo.header.stamp)
        t_radar = self._stamp_to_sec(self.latest_radar.header.stamp)

        # Stale kontrol (sim_time açıkken clock da sim time olur)
        if (now_sec - t_lidar) > self.max_age_sec:
            self.get_logger().warn(f"LiDAR stale: {now_sec - t_lidar:.2f}s", throttle_duration_sec=1)
            return
        if (now_sec - t_yolo) > self.max_age_sec:
            self.get_logger().warn(f"YOLO stale: {now_sec - t_yolo:.2f}s", throttle_duration_sec=1)
            return
        if (now_sec - t_radar) > self.max_age_sec:
            self.get_logger().warn(f"Radar stale: {now_sec - t_radar:.2f}s", throttle_duration_sec=1)
            return

        # ===== DRONE POZİSYONUNU AL (DRONE-RELATIVE BEV İÇİN) =====
        try:
            tf_base_odom = self.tf_buffer.lookup_transform(
                "odom",
                "base_link",
                rclpy.time.Time()
            )
            self.drone_x = tf_base_odom.transform.translation.x
            self.drone_y = tf_base_odom.transform.translation.y
            self.drone_z = tf_base_odom.transform.translation.z
        except Exception as e:
            self.get_logger().warn(f"Drone TF error: {e}")
            self.drone_x = 0.0
            self.drone_y = 0.0
            self.drone_z = 0.0

        # Boş BEV görüntüsü oluştur (siyah)
        bev = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        lidar_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        radar_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        yolo_layer  = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # 1. LiDAR (BEYAZ)
        self.draw_lidar(bev, lidar_layer)

        # 2. Radar (YEŞİL)
        self.draw_radar(bev, radar_layer)

        # 3. YOLO Projeksiyon (KIRMIZI) + ✅ publish projected detections
        self.draw_yolo_projections(bev, yolo_layer)

        # 4. Drone merkezi (MAVİ)
        cv2.circle(bev, (self.grid_size // 2, self.grid_size // 2), 4, (255, 0, 0), -1)

        # 5. Y ekseni düzeltmesi ve yayınlama
        bev = np.flipud(bev)
        lidar_layer = np.flipud(lidar_layer)
        radar_layer = np.flipud(radar_layer)
        yolo_layer  = np.flipud(yolo_layer)
        self.pub.publish(self.bridge.cv2_to_imgmsg(bev, encoding="bgr8"))
        self.pub_lidar.publish(self.bridge.cv2_to_imgmsg(lidar_layer, encoding="mono8"))
        self.pub_radar.publish(self.bridge.cv2_to_imgmsg(radar_layer, encoding="mono8"))
        self.pub_yolo_layer.publish(self.bridge.cv2_to_imgmsg(yolo_layer, encoding="mono8"))

    # ==================== LiDAR ====================
    def draw_lidar(self, bev, lidar_layer):
        try:
            tf_lidar = self.tf_buffer.lookup_transform(
                "odom",
                self.latest_lidar.header.frame_id,
                rclpy.time.Time()
            )
            points = pc2.read_points_list(
                self.latest_lidar,
                field_names=("x", "y", "z"),
                skip_nans=True
            )

            for p in points:
                pt = PointStamped()
                pt.point.x = float(p[0])
                pt.point.y = float(p[1])
                pt.point.z = float(p[2])

                g = tf2_geometry_msgs.do_transform_point(pt, tf_lidar)

                rel_x = g.point.x - self.drone_x
                rel_y = g.point.y - self.drone_y

                gx = int(rel_x / self.res + self.grid_size / 2)
                gy = int(rel_y / self.res + self.grid_size / 2)

                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    cv2.circle(bev, (gx, gy), 1, (255, 255, 255), -1)
                    cv2.circle(lidar_layer, (gx, gy), 2, 255, -1)

        except Exception as e:
            self.get_logger().debug(f"LiDAR TF error: {e}")

    # ==================== RADAR ====================
    def draw_radar(self, bev, radar_layer):
        if self.latest_radar is None:
            return

        try:
            tf_radar = self.tf_buffer.lookup_transform(
                "odom",
                self.latest_radar.header.frame_id,
                rclpy.time.Time()
            )

            for rp in self.latest_radar.points:
                pt = PointStamped()
                pt.point.x = float(rp.x)
                pt.point.y = float(rp.y)
                pt.point.z = float(rp.z)

                g = tf2_geometry_msgs.do_transform_point(pt, tf_radar)

                rel_x = g.point.x - self.drone_x
                rel_y = g.point.y - self.drone_y

                rx = int(rel_x / self.res + self.grid_size / 2)
                ry = int(rel_y / self.res + self.grid_size / 2)

                if 0 <= rx < self.grid_size and 0 <= ry < self.grid_size:
                    cv2.circle(bev, (rx, ry), 2, (0, 255, 0), -1)
                    cv2.circle(radar_layer, (rx, ry), 3, 255, -1)

        except Exception as e:
            self.get_logger().debug(f"Radar TF error: {e}")

    # ==================== YOLO PROJECTION + PUBLISH ====================
    def draw_yolo_projections(self, bev, yolo_layer):
        if self.latest_yolo is None:
            return

        # Timestamp (TF doğruluğu için)
        timestamp = self.latest_yolo.header.stamp

        # camera_link -> odom TF (o timestamp)
        try:
            tf_cam_odom = self.tf_buffer.lookup_transform(
                "odom",
                "camera_link",
                timestamp
            )
        except Exception as e:
            self.get_logger().warn(f"TF Lookup failed (camera_link->odom @stamp). Using latest. Error: {e}")
            try:
                tf_cam_odom = self.tf_buffer.lookup_transform("odom", "camera_link", rclpy.time.Time())
            except Exception as e2:
                self.get_logger().warn(f"TF Lookup failed (latest camera_link->odom): {e2}")
                return

        cam_x = tf_cam_odom.transform.translation.x
        cam_y = tf_cam_odom.transform.translation.y
        cam_z = tf_cam_odom.transform.translation.z

        IS_ON_GROUND = cam_z < 0.5
        VIRTUAL_HEIGHT = 0.24

        if IS_ON_GROUND:
            calc_h = VIRTUAL_HEIGHT
        else:
            calc_h = cam_z

        # ✅ NEW: projected detections message
        proj_array = Detection3DArray()
        proj_array.header.stamp = timestamp
        proj_array.header.frame_id = "base_link"  # we will publish base_link positions

        for det in self.latest_yolo.detections:
            # Bottom-center of bbox
            u = det.bbox.center.position.x
            v = det.bbox.center.position.y + (det.bbox.size_y / 2.0)

            # Normalize (pinhole)
            ray_x_norm = (u - self.cx) / self.fx
            ray_y_norm = (v - self.cy) / self.fy

            # Ray in camera frame
            ray_pt = PointStamped()
            ray_pt.header.frame_id = "camera_link"
            ray_pt.header.stamp = timestamp
            ray_pt.point.x = 1.0
            ray_pt.point.y = -ray_x_norm
            ray_pt.point.z = -ray_y_norm

            # Transform ray point to odom
            try:
                ray_odom = tf2_geometry_msgs.do_transform_point(ray_pt, tf_cam_odom)
            except Exception:
                continue

            dx = ray_odom.point.x - cam_x
            dy = ray_odom.point.y - cam_y
            dz = ray_odom.point.z - cam_z

            # Intersect with ground (Z=0)
            if dz >= -0.001:
                continue

            t = -calc_h / dz
            if t <= 0 or t > 100.0:
                continue

            ground_x = cam_x + t * dx
            ground_y = cam_y + t * dy

            # ----- DRONE-RELATIVE DRAW (for BEV image) -----
            rel_x = ground_x - self.drone_x
            rel_y = ground_y - self.drone_y

            bx = int(rel_x / self.res + self.grid_size / 2)
            by = int(rel_y / self.res + self.grid_size / 2)

            if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                cv2.circle(bev, (bx, by), 6, (0, 0, 255), -1)
                cv2.circle(bev, (bx, by), 7, (255, 255, 255), 1)
                cv2.circle(yolo_layer, (bx, by), 7, 255, -1)

            # ----- ✅ PUBLISH: convert ground point (odom) -> base_link -----
            gp = PointStamped()
            gp.header.stamp = timestamp
            gp.header.frame_id = "odom"
            gp.point.x = float(ground_x)
            gp.point.y = float(ground_y)
            gp.point.z = 0.0

            try:
                tf_base_odom_at_t = self.tf_buffer.lookup_transform(
                    "base_link",  # target
                    "odom",       # source
                    timestamp
                )
                gp_base = tf2_geometry_msgs.do_transform_point(gp, tf_base_odom_at_t)
            except Exception as e:
                self.get_logger().warn(f"YOLO ground->base_link TF failed: {e}")
                continue

            # Build Detection3D
            det3 = Detection3D()
            det3.header.stamp = timestamp
            det3.header.frame_id = "base_link"

            hypo = ObjectHypothesisWithPose()

            # Class + score from YOLO Detection2D
            if len(det.results) > 0:
                hypo.hypothesis.class_id = det.results[0].hypothesis.class_id
                hypo.hypothesis.score = float(det.results[0].hypothesis.score)
            else:
                hypo.hypothesis.class_id = "-1"
                hypo.hypothesis.score = 0.0

            # Put projected point into pose
            hypo.pose.pose.position.x = float(gp_base.point.x)
            hypo.pose.pose.position.y = float(gp_base.point.y)
            hypo.pose.pose.position.z = 0.0
            hypo.pose.pose.orientation.w = 1.0

            det3.results.append(hypo)
            proj_array.detections.append(det3)

        # ✅ publish if any
        if len(proj_array.detections) > 0:
            self.pub_yolo_proj.publish(proj_array)


def main():
    rclpy.init()
    node = BEVImageNodeV2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()