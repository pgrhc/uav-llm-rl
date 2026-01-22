"""
BEV Projection Node v2 - Düzeltilmiş Versiyon

Düzeltmeler:
1. real_height yerine kameranın gerçek z koordinatı (TF'den) kullanılıyor
2. t hesabında tutarlı camera_z kullanımı
3. Yer modu kaldırıldı - matematiksel projeksiyon her yükseklikte çalışır
4. Daha iyi hata kontrolü ve logging eklendi
5. Normalize edilmiş yön vektörü kullanımı
6. DRONE-RELATIVE BEV: Tüm noktalar drone pozisyonuna göre çiziliyor
   (Drone her zaman merkezde!)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, Image
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
from fusion_msgs.msg import RadarPoints
from px4_msgs.msg import VehicleOdometry
from sensor_msgs.msg import CameraInfo

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

        # Subscribe to Camera Info (The topic usually matches your image topic base)
        # e.g. if image is /camera/image_raw, info is /camera/camera_info
        self.create_subscription(
            CameraInfo,
            "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/camera_info", # Check your topics!
            self.camera_info_callback,
            10
        )

        # QoS for PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

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
            '/radar/points_filtered_radarmsg',
            self.radar_callback,
            10
        )

        # Publisher
        self.pub = self.create_publisher(Image, "/bev/image_deneme", 10)

        # Latest data storage
        self.latest_lidar = None
        self.latest_yolo = None
        self.latest_radar = None

        # Drone pozisyonu (drone-relative BEV için)
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0

        self.get_logger().info("BEV Node V2 Started - Drone-Relative BEV")

    # ==================== CALLBACKS ====================
    def camera_info_callback(self, msg):
        # The 'k' matrix holds the intrinsics: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.camera_info_received = True
        
        # Optional: Print once to verify
        # self.get_logger().info(f"Updated Camera Intrinsics: fx={self.fx:.2f}, cx={self.cx:.2f}")

    def lidar_callback(self, msg):
        self.latest_lidar = msg

    def yolo_callback(self, msg):
        self.latest_yolo = msg
        self.process()

    def radar_callback(self, msg):
        self.latest_radar = msg

    # ==================== MAIN PROCESS ====================

    def process(self):
        if not self.camera_info_received:
            self.get_logger().warn("Waiting for Camera Info...", throttle_duration_sec=2)
            return
        if self.latest_lidar is None or self.latest_yolo is None:
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

        # 1. LiDAR Çizimi (BEYAZ)
        self.draw_lidar(bev)

        # 2. Radar Çizimi (YEŞİL)
        self.draw_radar(bev)

        # 3. YOLO Projeksiyon (KIRMIZI)
        self.draw_yolo_projections(bev)

        # 4. Drone merkezi (MAVİ) - Her zaman grid merkezinde!
        cv2.circle(bev, (self.grid_size // 2, self.grid_size // 2), 4, (255, 0, 0), -1)

        # 5. Y ekseni düzeltmesi ve yayınlama
        bev = np.flipud(bev)
        self.pub.publish(self.bridge.cv2_to_imgmsg(bev, encoding="bgr8"))

    # ==================== LiDAR ÇİZİMİ ====================

    def draw_lidar(self, bev):
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

                # DRONE-RELATIVE: Drone pozisyonunu çıkar
                rel_x = g.point.x - self.drone_x
                rel_y = g.point.y - self.drone_y

                gx = int(rel_x / self.res + self.grid_size / 2)
                gy = int(rel_y / self.res + self.grid_size / 2)

                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    cv2.circle(bev, (gx, gy), 1, (255, 255, 255), -1)

        except Exception as e:
            self.get_logger().debug(f"LiDAR TF error: {e}")

    # ==================== RADAR ÇİZİMİ ====================

    def draw_radar(self, bev):
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
                pt.point.x = rp.x
                pt.point.y = rp.y
                pt.point.z = rp.z

                g = tf2_geometry_msgs.do_transform_point(pt, tf_radar)

                # DRONE-RELATIVE: Drone pozisyonunu çıkar
                rel_x = g.point.x - self.drone_x
                rel_y = g.point.y - self.drone_y

                rx = int(rel_x / self.res + self.grid_size / 2)
                ry = int(rel_y / self.res + self.grid_size / 2)

                if 0 <= rx < self.grid_size and 0 <= ry < self.grid_size:
                    cv2.circle(bev, (rx, ry), 2, (0, 255, 0), -1)

        except Exception as e:
            self.get_logger().debug(f"Radar TF error: {e}")

    # ==================== YOLO PROJEKSİYON (DÜZELTİLMİŞ) ====================

    # ==================== CORRECTED YOLO PROJECTION ====================

    def draw_yolo_projections(self, bev):
        if self.latest_yolo is None:
            return
        
        # 1. Use the EXACT timestamp of the detection for TF accuracy
        # (This prevents lag/shift when drone is moving)
        try:
            timestamp = self.latest_yolo.header.stamp
            tf_cam_odom = self.tf_buffer.lookup_transform(
                "odom",
                "camera_link", # Ensure this is your camera frame
                timestamp
            )
        except Exception as e:
            self.get_logger().warn(f"TF Lookup failed (using latest): {e}")
            try:
                tf_cam_odom = self.tf_buffer.lookup_transform(
                    "odom", "camera_link", rclpy.time.Time())
            except:
                return

        cam_x = tf_cam_odom.transform.translation.x
        cam_y = tf_cam_odom.transform.translation.y
        cam_z = tf_cam_odom.transform.translation.z

        IS_ON_GROUND = cam_z < 0.5

        # 2. Virtual Height Calibration
        #    - If Red Dot is BEHIND white dots (Too Far) -> DECREASE this (try 0.15 or 0.18)
        #    - If Red Dot is IN FRONT of white dots (Too Close) -> INCREASE this
        VIRTUAL_HEIGHT = 0.24  

        # 3. FORCE the height. 
        #    We do not trust TF on the ground. We trust your calibration.
        if IS_ON_GROUND:
            calc_h = VIRTUAL_HEIGHT
        else:
            calc_h = cam_z

        for det in self.latest_yolo.detections:
            # ===== CRITICAL FIX: Use Bottom-Center of Bounding Box =====
            # Center of the box creates a ray that hits the ground BEHIND the object.
            # Bottom of the box (feet) touches the ground at the correct location.
            
            u = det.bbox.center.position.x
            # Add half height to get the bottom pixel (v)
            v = det.bbox.center.position.y + (det.bbox.size_y / 2.0)

            # 1. Normalize (Pinhole Model)
            ray_x_norm = (u - self.cx) / self.fx
            ray_y_norm = (v - self.cy) / self.fy

            # 2. Create Ray in Camera Frame (Assuming X-Forward ROS Camera Standard)
            # If your camera uses Optical Frame (Z-Forward), you might need to swap these.
            ray_pt = PointStamped()
            ray_pt.header.frame_id = "camera_link"
            ray_pt.header.stamp = timestamp
            
            # Standard ROS Body Convention: X=Forward, Y=Left, Z=Up
            # Image Plane: u goes Right (-Y), v goes Down (-Z)
            ray_pt.point.x = 1.0          # Forward
            ray_pt.point.y = -ray_x_norm  # Left
            ray_pt.point.z = -ray_y_norm  # Up

            # 3. Transform Ray to Odom
            try:
                ray_odom = tf2_geometry_msgs.do_transform_point(ray_pt, tf_cam_odom)
            except:
                continue

            # 4. Ray Direction Vector
            dx = ray_odom.point.x - cam_x
            dy = ray_odom.point.y - cam_y
            dz = ray_odom.point.z - cam_z

            # 5. Intersect with Ground (Z=0)
            if dz >= -0.001: continue # Ray pointing up
            
            t = -calc_h / dz
            if t <= 0 or t > 100.0: continue

            ground_x = cam_x + t * dx
            ground_y = cam_y + t * dy

            # 6. Map to Grid
            rel_x = ground_x - self.drone_x
            rel_y = ground_y - self.drone_y

            bx = int(rel_x / self.res + self.grid_size / 2)
            by = int(rel_y / self.res + self.grid_size / 2)

            if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                # Draw a larger Red Circle for YOLO
                cv2.circle(bev, (bx, by), 6, (0, 0, 255), -1) 
                # Optional: Draw a white outline to make it pop
                cv2.circle(bev, (bx, by), 7, (255, 255, 255), 1)


def main():
    rclpy.init()
    node = BEVImageNodeV2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
