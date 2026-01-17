import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy # <--- EKLENDI
from sensor_msgs.msg import PointCloud2, Image
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
from fusion_msgs.msg import RadarPoints
from transforms3d.quaternions import quat2mat

# <--- EKLENDI: PX4 Mesajı
from px4_msgs.msg import VehicleOdometry 

import numpy as np
import cv2

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
import math

class BEVImageNode(Node):
    def __init__(self):
        super().__init__("bev_image_node")

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # BEV grid
        self.grid_size = 600
        self.res = 0.05  # m/cell

        # --- Kamera intrinsics ---
        self.fx = 539.9363
        self.fy = 539.9363
        self.cx = 640.0
        self.cy = 480.0

        # <--- EKLENDI: Yükseklik Değişkeni
        self.current_height = 0.0

        # <--- EKLENDI: PX4 Odometry Aboneliği (QoS Ayarlı)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(
            VehicleOdometry, 
            "/fmu/out/vehicle_odometry", 
            self.odom_callback, 
            qos_profile
        )

        # Subs
        self.create_subscription(
            PointCloud2,
            "/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan/points",
            self.lidar_callback,
            10,
        )

        self.create_subscription(
            Detection2DArray,
            "/yolo/detections",
            self.yolo_callback,
            10,
        )
        self.create_subscription(
            RadarPoints,
            '/radar/points_filtered_radarmsg',
            self.radar_callback,
            10
        )

        # Pub
        self.pub = self.create_publisher(Image, "/bev/image", 10)

        self.latest_lidar = None
        self.latest_yolo = None
        self.latest_radar = None

        self.get_logger().info("BEV IMAGE NODE STARTED - REAL HEIGHT MODE ACTIVE")

    # <--- EKLENDI: Yükseklik Callback'i
    def odom_callback(self, msg):
        # PX4 Z ekseni aşağı negatiftir. Yükseklik için tersini alıyoruz.
        self.current_height = -1.0 * msg.position[2]

    # ---------------- LIDAR ----------------
    def lidar_callback(self, msg: PointCloud2):
        self.latest_lidar = msg

    # ---------------- YOLO ----------------
    def yolo_callback(self, msg: Detection2DArray):
        self.latest_yolo = msg
        self.process()

    def radar_callback(self, msg: RadarPoints):
        self.latest_radar = msg
        self.process()

    # ------------- ORTAK İŞLEME -------------
    def process(self):
        if self.latest_lidar is None or self.latest_yolo is None:
            return

        bev = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # ====== 1) LIDAR BEV ÇİZİMİ ======
        points = pc2.read_points_list(
            self.latest_lidar,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )

        try:
            tf_lidar_to_odom = self.tf_buffer.lookup_transform(
                "odom",
                self.latest_lidar.header.frame_id,
                rclpy.time.Time(),
            )
        except Exception:
            return

        for p in points:
            pt = PointStamped()
            pt.header.frame_id = self.latest_lidar.header.frame_id
            pt.point.x = float(p[0])
            pt.point.y = float(p[1])
            pt.point.z = float(p[2])

            try:
                g = tf2_geometry_msgs.do_transform_point(pt, tf_lidar_to_odom)
            except Exception:
                continue

            gx = int(g.point.x / self.res + self.grid_size / 2)
            gy = int(g.point.y / self.res + self.grid_size / 2)

            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                cv2.circle(bev, (gx, gy), 2, (255, 255, 255), -1)
        
        # ====== RADAR ÇİZİMİ ======
        if self.latest_radar is not None:
            try:
                tf_radar_to_odom = self.tf_buffer.lookup_transform(
                    "odom",
                    self.latest_radar.header.frame_id,
                    rclpy.time.Time()
                )
                
                for rp in self.latest_radar.points:
                    pt_r = PointStamped()
                    pt_r.header.frame_id = self.latest_radar.header.frame_id
                    pt_r.point.x = rp.x
                    pt_r.point.y = rp.y
                    pt_r.point.z = rp.z

                    g_r = tf2_geometry_msgs.do_transform_point(pt_r, tf_radar_to_odom)

                    rx = int(g_r.point.x / self.res + self.grid_size / 2)
                    ry = int(g_r.point.y / self.res + self.grid_size / 2)

                    if 0 <= rx < self.grid_size and 0 <= ry < self.grid_size:
                        cv2.circle(bev, (rx, ry), 2, (0, 255, 0), -1) 

            except Exception as e:
                pass

        # ====== 2) YOLO DETEKSIYONLARINI BEV'E PROJE ET ======
        try:
            tf_cam_to_odom = self.tf_buffer.lookup_transform(
                "odom",       # target
                "camera_link",     # source
                rclpy.time.Time(),
            )
        except Exception:
            img_msg = self.bridge.cv2_to_imgmsg(bev, encoding="bgr8")
            self.pub.publish(img_msg)
            return

        o = np.array([
            tf_cam_to_odom.transform.translation.x,
            tf_cam_to_odom.transform.translation.y,
            tf_cam_to_odom.transform.translation.z
        ], dtype=np.float64)

        # <--- EKLENDI: Yükseklik Düzeltmesi
        # Eğer PX4'ten mantıklı bir yükseklik geliyorsa, TF'in bozuk Z'sini ez.
        if self.current_height > 0.2:
            o[2] = self.current_height
        # Eğer veri gelmediyse ama TF çok düşükse (0.24m gibi) manuel yükselt
        elif o[2] < 0.5:
             o[2] = 5.0 # Varsayılan uçuş yüksekliği

        q = tf_cam_to_odom.transform.rotation
        R = quat2mat([q.w, q.x, q.y, q.z])

        for det in self.latest_yolo.detections:
            u = det.bbox.center.position.x
            v = det.bbox.center.position.y

            # Optical ray
            rx = (u - self.cx) / self.fx
            ry = (v - self.cy) / self.fy

            # Optical -> camera_link (senin varsayımın)
            # camera_link: X ileri, Y sol, Z yukarı
            d_cam = np.array([1.0, -rx, -ry], dtype=np.float64)

            # TARGET_FRAME'e döndür (SADECE rotasyon!)
            d = R @ d_cam

            # Yer düzlemi z=0 (TARGET_FRAME'de)
            if d[2] >= -1e-3:   # yukarı/ufka bakıyorsa
                continue

            t = -o[2] / d[2]
            if t <= 0:
                continue
            
            # Kalibrasyon Çarpanı (İsteğe bağlı, cisimler çok yakın görünüyorsa aç)
            # t = t * 1.1 

            P = o + t * d   # [Xw, Yw, Zw≈0]

            Xw, Yw = P[0], P[1]

            bx = int(Xw / self.res + self.grid_size / 2)
            by = int(Yw / self.res + self.grid_size / 2)

            if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                cv2.circle(bev, (bx, by), 5, (0, 0, 255), -1)                
        
        # Drone merkezi ve flip işlemi
        cv2.circle(bev, (self.grid_size // 2, self.grid_size // 2),
                   4, (255, 0, 0), -1)   
        
        # Eğer görüntün ters ise bunu kullan, düz ise sil
        bev = np.flipud(bev)

        img_msg = self.bridge.cv2_to_imgmsg(bev, encoding="bgr8")
        self.pub.publish(img_msg)


def main():
    rclpy.init()
    node = BEVImageNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()