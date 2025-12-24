import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
from fusion_msgs.msg import RadarPoints

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
        

        # BEV grid
        self.grid_size = 600
        self.res = 0.05  # m/cell

        # --- Kamera intrinsics (CameraInfo'dan okuduğun değerler) ---
        # K matrisi:
        # [ 539.93   0     640 ]
        # [   0    539.93  480 ]
        # [   0      0      1  ]
        self.fx = 539.9363
        self.fy = 539.9363
        self.cx = 640.0
        self.cy = 480.0

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

        self.get_logger().info("BEV IMAGE NODE STARTED")

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
                "base_link",
                self.latest_lidar.header.frame_id,
                rclpy.time.Time(),
            )
        except Exception:
            # TF henüz hazır değilse
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
        
        if self.latest_radar is not None:
            try:
                # Radarın frame_id'sinden odom'a dönüşüm al (örn: radar_link -> odom)
                tf_radar_to_odom = self.tf_buffer.lookup_transform(
                    "base_link",
                    self.latest_radar.header.frame_id,
                    rclpy.time.Time()
                )
                
                # Radar noktalarını döngüye al
                for rp in self.latest_radar.points:
                    # RadarPoint msg -> PointStamped
                    pt_r = PointStamped()
                    pt_r.header.frame_id = self.latest_radar.header.frame_id
                    pt_r.point.x = rp.x
                    pt_r.point.y = rp.y
                    pt_r.point.z = rp.z

                    # Transform
                    g_r = tf2_geometry_msgs.do_transform_point(pt_r, tf_radar_to_odom)

                    # Grid'e maple
                    rx = int(g_r.point.x / self.res + self.grid_size / 2)
                    ry = int(g_r.point.y / self.res + self.grid_size / 2)

                    # Çizim (YEŞİL - BGR Formatında (0, 255, 0))
                    # Radarı biraz daha büyük çiziyoruz ki fark edilsin (Radius: 3)
                    if 0 <= rx < self.grid_size and 0 <= ry < self.grid_size:
                        cv2.circle(bev, (rx, ry), 1, (0, 255, 0), -1) 

            except Exception as e:
                # TF hatası olursa radarı atla, diğerlerini çizmeye devam et
                pass

        # ========== 2) YOLO DETEKSIYONLARINI BEV'E PROJE ET ==========
        try:
            tf_cam_to_odom = self.tf_buffer.lookup_transform(
                "base_link",          # target
                "camera_link",   # source
                rclpy.time.Time(),
            )
        except Exception:
            # kamera TF'i yoksa sadece lidar çiz
            img_msg = self.bridge.cv2_to_imgmsg(bev, encoding="bgr8")
            self.pub.publish(img_msg)
            return

        # Kamera orijini (odom frame'de)
        cam_origin_cam = PointStamped()
        cam_origin_cam.header.frame_id = "camera_link"
        cam_origin_cam.point.x = 0.0
        cam_origin_cam.point.y = 0.0
        cam_origin_cam.point.z = 0.0
        cam_origin_odom = tf2_geometry_msgs.do_transform_point(
            cam_origin_cam, tf_cam_to_odom
        )

        x0 = tf_cam_to_odom.transform.translation.x
        y0 = tf_cam_to_odom.transform.translation.y
        z0 = tf_cam_to_odom.transform.translation.z

        for det in self.latest_yolo.detections:
            # Tip: Objenin 'ayak' bastığı yeri bulmak için bottom-center kullanıyoruz
            u = det.bbox.center.position.x
            v = det.bbox.center.position.y + (det.bbox.size_y / 2.0) # Alt kenar

            # 1. Pixel -> Camera Ray (Standard Pinhole Model)
            # ROS Camera Frame: Z forward, X right, Y down
            ray_x = (u - self.cx) / self.fx
            ray_y = (v - self.cy) / self.fy
            ray_z = 1.0 

            # 2. Ray'i Odom frame'ine çevir
            # Ray ucunda bir nokta oluşturup transform ediyoruz
            pt_ray_cam = PointStamped()
            pt_ray_cam.header.frame_id = "camera_link"
            pt_ray_cam.point.x = ray_x
            pt_ray_cam.point.y = ray_y
            pt_ray_cam.point.z = ray_z

            try:
                pt_ray_odom = tf2_geometry_msgs.do_transform_point(pt_ray_cam, tf_cam_to_odom)
            except Exception:
                continue

            # Ray vektörü (Odom frame'de)
            dx = pt_ray_odom.point.x - x0
            dy = pt_ray_odom.point.y - y0
            dz = pt_ray_odom.point.z - z0

            # 3. Yer düzlemi (Z=0) ile kesişim
            # Denklem: z0 + s * dz = 0  =>  s = -z0 / dz
            if abs(dz) < 1e-4: # Ray yere paralel ise atla
                continue

            s = -z0 / dz
            
            # Sadece drone'un önündeki ve makul uzaklıktaki noktaları çiz
            if s > 0:
                xw = x0 + s * dx
                yw = y0 + s * dy

                bx = int(xw / self.res + self.grid_size / 2)
                by = int(yw / self.res + self.grid_size / 2)

                if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                    # Daha belirgin olması için 5px kırmızı daire
                    cv2.circle(bev, (bx, by), 5, (0, 0, 255), -1)
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