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

import numpy as np
import cv2
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

class BEVImageNode(Node):
    def __init__(self):
        super().__init__("bev_image_node")

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.grid_size = 600
        self.res = 0.05  
        self.fx = 539.9363
        self.fy = 539.9363
        self.cx = 640.0
        self.cy = 480.0

        self.current_height = 0.0

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.odom_callback, qos_profile)

        self.create_subscription(PointCloud2, "/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan/points", self.lidar_callback, 10)
        self.create_subscription(Detection2DArray, "/yolo/detections", self.yolo_callback, 10)
        self.create_subscription(RadarPoints, '/radar/points_filtered_radarmsg', self.radar_callback, 10)

        self.pub = self.create_publisher(Image, "/bev/image", 10)
        self.latest_lidar = None
        self.latest_yolo = None
        self.latest_radar = None

        self.get_logger().info("BEV NODE: LOW ALTITUDE SAFE MODE ACTIVE")

    def odom_callback(self, msg):
        self.current_height = -1.0 * msg.position[2]

    def lidar_callback(self, msg): self.latest_lidar = msg
    def yolo_callback(self, msg): 
        self.latest_yolo = msg
        self.process()
    def radar_callback(self, msg): self.latest_radar = msg

    def process(self):
        if self.latest_lidar is None or self.latest_yolo is None: return

        bev = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # 1. LIDAR ÇİZİMİ
        try:
            tf_lidar = self.tf_buffer.lookup_transform("odom", self.latest_lidar.header.frame_id, rclpy.time.Time())
            points = pc2.read_points_list(self.latest_lidar, field_names=("x", "y", "z"), skip_nans=True)
            for p in points:
                pt = PointStamped()
                pt.point.x, pt.point.y, pt.point.z = float(p[0]), float(p[1]), float(p[2])
                g = tf2_geometry_msgs.do_transform_point(pt, tf_lidar)
                gx, gy = int(g.point.x/self.res + self.grid_size/2), int(g.point.y/self.res + self.grid_size/2)
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    cv2.circle(bev, (gx, gy), 1, (255, 255, 255), -1)
        except: pass

        # 2. RADAR ÇİZİMİ
        if self.latest_radar:
            try:
                tf_radar = self.tf_buffer.lookup_transform("odom", self.latest_radar.header.frame_id, rclpy.time.Time())
                for rp in self.latest_radar.points:
                    pt = PointStamped()
                    pt.point.x, pt.point.y, pt.point.z = rp.x, rp.y, rp.z
                    g = tf2_geometry_msgs.do_transform_point(pt, tf_radar)
                    rx, ry = int(g.point.x/self.res + self.grid_size/2), int(g.point.y/self.res + self.grid_size/2)
                    if 0 <= rx < self.grid_size and 0 <= ry < self.grid_size:
                        cv2.circle(bev, (rx, ry), 2, (0, 255, 0), -1)
            except: pass

        # 3. YOLO (GÜÇLENDİRİLMİŞ YER MODU)
        try:
            tf_cam_odom = self.tf_buffer.lookup_transform("odom", "camera_link", rclpy.time.Time())
        except:
            self.pub.publish(self.bridge.cv2_to_imgmsg(bev, encoding="bgr8"))
            return

        # Yükseklik Kontrolü
        real_height = self.current_height
        is_on_ground = False
        
        # Eğer yükseklik 0.4m altındaysa "YER MODU"nu aç
        if real_height < 0.4:
            is_on_ground = True
            real_height = max(real_height, 0.2) # Matematik patlamasın diye min 0.2 al

        for det in self.latest_yolo.detections:
            u = det.bbox.center.position.x
            v = det.bbox.center.position.y

            # 1. Işın Oluştur
            ray_z = 1.0
            ray_x = (u - self.cx) / self.fx
            ray_y = (v - self.cy) / self.fy

            # 2. Frame Çevir
            ray_pt = PointStamped()
            ray_pt.header.frame_id = "camera_link"
            ray_pt.point.x = ray_z    
            ray_pt.point.y = -ray_x   
            ray_pt.point.z = -ray_y   

            try:
                ray_odom = tf2_geometry_msgs.do_transform_point(ray_pt, tf_cam_odom)
            except: continue

            # Kamera Merkezi
            ox = tf_cam_odom.transform.translation.x
            oy = tf_cam_odom.transform.translation.y
            
            # Yön Vektörü
            dx = ray_odom.point.x - ox
            dy = ray_odom.point.y - oy
            dz = ray_odom.point.z - tf_cam_odom.transform.translation.z

            t = 0.0

            # --- MANTIK: HAVADA MIYIZ YERDE MİYİZ? ---
            
            if is_on_ground:
                # *** YER MODU AKTİF ***
                # Matematiksel kesişim yerine "Sezgisel Mesafe" kullanacağız.
                # Ekranda (v) ne kadar aşağıdaysa o kadar yakındır.
                # v=480 (en alt) -> 1 metre
                # v=240 (orta)   -> 10 metre (Ufuk)
                
                # Basit bir Mapping:
                H = int(2.0 * self.cy)  # 960
                pixel_from_bottom = H - v

                # negatif/0 olmasın
                if pixel_from_bottom < 10.0:
                    pixel_from_bottom = 10.0

                t = (real_height * self.fy) / pixel_from_bottom

                # güvenli aralık (yerdeyken çok yakına da çok uzağa da gitmesin)
                if t < 0.5:
                    t = 0.5
                if t > 15.0:
                    t = 15.0
                
            else:
                # *** UÇUŞ MODU (Klasik Matematik) ***
                if dz >= -0.01: continue # Işın havaya bakıyor
                t = -real_height / dz
                if t <= 0: continue

            # Koordinat Hesabı
            Xw = ox + t * dx
            Yw = oy + t * dy

            bx = int(Xw / self.res + self.grid_size / 2)
            by = int(Yw / self.res + self.grid_size / 2)

            if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                cv2.circle(bev, (bx, by), 5, (0, 0, 255), -1)

        cv2.circle(bev, (self.grid_size//2, self.grid_size//2), 4, (255, 0, 0), -1)
        bev = np.flipud(bev)
        self.pub.publish(self.bridge.cv2_to_imgmsg(bev, encoding="bgr8"))

def main():
    rclpy.init()
    node = BEVImageNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()