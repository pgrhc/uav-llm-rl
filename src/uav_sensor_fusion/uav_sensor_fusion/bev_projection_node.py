import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection3DArray, Detection3D, ObjectHypothesisWithPose

from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
from fusion_msgs.msg import RadarPoints

import numpy as np
import cv2
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation


class BEVImageNodeV2(Node):
    def __init__(self):
        super().__init__("bev_image_node_v2")

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # BEV Grid Parametreleri
        self.grid_size = 600
        self.res = 80 / self.grid_size

        # Kamera Intrinsics
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

        self.pub = self.create_publisher(Image, "/bev/image", 10)
        self.pub_yolo_proj = self.create_publisher(
            Detection3DArray,
            "/yolo/projected_detections",
            10
        )

        # Latest data storage
        self.latest_lidar = None
        self.latest_yolo = None
        self.latest_radar = None

        # Drone pozisyonu
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0

        self.max_age_sec = 0.5
        self.create_timer(0.1, self.process)

        self.get_logger().info("BEV Node V2 Started - Adaptive Camera Pitch")

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

        if (now_sec - t_lidar) > self.max_age_sec:
            self.get_logger().warn(f"LiDAR stale: {now_sec - t_lidar:.2f}s", throttle_duration_sec=1)
            return
        if (now_sec - t_yolo) > self.max_age_sec:
            self.get_logger().warn(f"YOLO stale: {now_sec - t_yolo:.2f}s", throttle_duration_sec=1)
            return
        if (now_sec - t_radar) > self.max_age_sec:
            self.get_logger().warn(f"Radar stale: {now_sec - t_radar:.2f}s", throttle_duration_sec=1)
            return

        # Drone pozisyonu
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

        # Boş BEV
        bev = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        lidar_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        radar_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        yolo_layer  = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # 1. LiDAR
        self.draw_lidar(bev, lidar_layer)

        # 2. Radar
        self.draw_radar(bev, radar_layer)

        # 3. YOLO (FIX: Adaptive)
        self.draw_yolo_projections(bev, yolo_layer)

        # 4. Drone
        cv2.circle(bev, (self.grid_size // 2, self.grid_size // 2), 4, (255, 0, 0), -1)

        # 5. Flip & Publish
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

    # ==================== YOLO PROJECTION (FIXED) ====================
    def draw_yolo_projections(self, bev, yolo_layer):
        if self.latest_yolo is None:
            return

        timestamp = self.latest_yolo.header.stamp

        try:
            tf_cam_odom = self.tf_buffer.lookup_transform(
                "odom", "camera_link", rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except Exception as e:
            self.get_logger().warn(f"TF failed: {e}", throttle_duration_sec=1.0)
            return

        cam_x = tf_cam_odom.transform.translation.x
        cam_y = tf_cam_odom.transform.translation.y
        cam_z = tf_cam_odom.transform.translation.z

        # ✅ KAMERA PITCH AÇISINI AL
        q = tf_cam_odom.transform.rotation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        euler = rot.as_euler('xyz', degrees=True)
        pitch = euler[1]  # Y eksenindeki rotasyon (pitch)

        # ✅ PITCH'E GÖRE ADAPTIVE PARAMETRELER
        # Negatif pitch = aşağı bakıyor
        # Pozitif pitch = yukarı bakıyor
        # 0 pitch = yatay
        
        if cam_z < 0.5:  # YERDE
            # Yataya yakın kamera, minimal filtre
            min_pitch_deg = -5.0  # En az 5° aşağı bakmalı (esnek)
            max_proj_dist = 25.0
            tolerance = 0.1  # Geniş tolerans
            
        elif cam_z < 3.0:  # DÜŞÜK İRTİFA
            min_pitch_deg = -10.0  # En az 10° aşağı
            max_proj_dist = cam_z * 10.0
            tolerance = 0.05
            
        else:  # YÜKSEK İRTİFA
            min_pitch_deg = -20.0  # En az 20° aşağı
            max_proj_dist = cam_z * 12.0
            tolerance = 0.02
        max_proj_dist = min(max_proj_dist, 100.0)
        # ✅ DEBUG: Kamera durumunu logla
        self.get_logger().info(
            f"Camera: alt={cam_z:.2f}m, pitch={pitch:.1f}°, "
            f"min_pitch={min_pitch_deg:.1f}°, max_dist={max_proj_dist:.1f}m",
            throttle_duration_sec=2.0
        )

        # ✅ PITCH KONTROLÜ
        if pitch > min_pitch_deg:
            self.get_logger().warn(
                f"Camera not looking down enough! pitch={pitch:.1f}° > {min_pitch_deg:.1f}°",
                throttle_duration_sec=2.0
            )
            # Devam et ama daha toleranslı ol
            tolerance *= 2.0

        proj_array = Detection3DArray()
        proj_array.header.stamp = timestamp
        proj_array.header.frame_id = "base_link"

        debug_count = 0
        debug_rejected = {"pitch": 0, "dz": 0, "t": 0, "dist": 0, "bounds": 0}

        for det in self.latest_yolo.detections:
            # Bbox alt kenarı
            u = det.bbox.center.position.x
            v = det.bbox.center.position.y + (det.bbox.size_y / 2.0)

            # Normalized ray direction (camera frame)
            ray_x_norm = (u - self.cx) / self.fx
            ray_y_norm = (v - self.cy) / self.fy

            # Ray in camera frame
            ray_pt = PointStamped()
            ray_pt.header.frame_id = "camera_link"
            ray_pt.header.stamp = timestamp
            ray_pt.point.x = 1.0
            ray_pt.point.y = -ray_x_norm
            ray_pt.point.z = -ray_y_norm

            # Transform to odom
            try:
                ray_odom = tf2_geometry_msgs.do_transform_point(ray_pt, tf_cam_odom)
            except Exception:
                continue

            # Ray direction
            dx = ray_odom.point.x - cam_x
            dy = ray_odom.point.y - cam_y
            dz = ray_odom.point.z - cam_z

            # ✅ ESNEK DZ KONTROLÜ
            # dz < 0 olmalı (aşağı bakıyor)
            # Ama tolerance kadar esneklik ver
            if dz > tolerance:
                debug_rejected["dz"] += 1
                continue

            # Zemine kesişme parametresi
            # z = cam_z + t * dz = 0  →  t = -cam_z / dz
            if abs(dz) < 1e-6:  # Çok yatay ray
                debug_rejected["dz"] += 1
                continue

            t = -cam_z / dz

            # t pozitif ve makul olmalı
            if t <= 0 or t > max_proj_dist:
                debug_rejected["t"] += 1
                continue

            # Zemin noktası
            ground_x = cam_x + t * dx
            ground_y = cam_y + t * dy

            # Drone-relative
            rel_x = ground_x - self.drone_x
            rel_y = ground_y - self.drone_y
            rel_dist = float(np.hypot(rel_x, rel_y))

            # Distance check
            if rel_dist > max_proj_dist:
                debug_rejected["dist"] += 1
                continue

            # BEV grid
            bx = int(rel_x / self.res + self.grid_size / 2)
            by = int(rel_y / self.res + self.grid_size / 2)

            if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                cv2.circle(bev, (bx, by), 6, (0, 0, 255), -1)
                cv2.circle(bev, (bx, by), 7, (255, 255, 255), 1)
                cv2.circle(yolo_layer, (bx, by), 7, 255, -1)
                debug_count += 1
            else:
                debug_rejected["bounds"] += 1
                continue

            # ✅ 3D Detection message
            gp = PointStamped()
            gp.header.stamp = timestamp
            gp.header.frame_id = "odom"
            gp.point.x = float(ground_x)
            gp.point.y = float(ground_y)
            gp.point.z = 0.0

            try:
                tf_base_odom_at_t = self.tf_buffer.lookup_transform(
                    "base_link", "odom", rclpy.time.Time()
                )
                gp_base = tf2_geometry_msgs.do_transform_point(gp, tf_base_odom_at_t)
            except Exception:
                continue

            det3 = Detection3D()
            det3.header.stamp = timestamp
            det3.header.frame_id = "base_link"
            
            hypo = ObjectHypothesisWithPose()
            if len(det.results) > 0:
                hypo.hypothesis.class_id = det.results[0].hypothesis.class_id
                hypo.hypothesis.score = float(det.results[0].hypothesis.score)
            else:
                hypo.hypothesis.class_id = "-1"
                hypo.hypothesis.score = 0.0
                
            hypo.pose.pose.position.x = float(gp_base.point.x)
            hypo.pose.pose.position.y = float(gp_base.point.y)
            hypo.pose.pose.position.z = 0.0
            hypo.pose.pose.orientation.w = 1.0
            
            det3.results.append(hypo)
            proj_array.detections.append(det3)

        # ✅ DETAILED DEBUG LOG
        total_dets = len(self.latest_yolo.detections)
        if total_dets > 0:
            self.get_logger().info(
                f"YOLO: total={total_dets} | projected={debug_count} | "
                f"rejected: pitch={debug_rejected['pitch']}, dz={debug_rejected['dz']}, "
                f"t={debug_rejected['t']}, dist={debug_rejected['dist']}, "
                f"bounds={debug_rejected['bounds']}",
                throttle_duration_sec=1.0
            )

        # Publish
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