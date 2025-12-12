import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2  # ROS2'de mevcut
import numpy as np


class LidarDebug(Node):
    def __init__(self):
        super().__init__("lidar_debug")

        self.sub = self.create_subscription(
            PointCloud2,
            "/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan/points",
            self.cb,
            10,
        )

        # Sadece ilk taramayı kaydetmek için flag
        self.first_scan_saved = False

        # Kaydedeceğimiz dosya yolu (istersen değiştir)
        self.output_path = "/home/ubuntu/Desktop/ros2_env/first_lidar_scan.npy"

        self.get_logger().info(
            "LidarDebug node started, waiting for first PointCloud2 scan..."
        )

    def cb(self, msg: PointCloud2):
        if self.first_scan_saved:
            # İlk taramayı kaydettik, diğerlerini yok say
            return

        self.get_logger().info("Received first LiDAR scan, converting to numpy...")

        # Tüm noktaları (x, y, z) olarak oku
        points_iter = point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True
        )

        points_list = []
        for p in points_iter:
            # p = (x, y, z)
            points_list.append([p[0], p[1], p[2]])

        if not points_list:
            self.get_logger().warn("No points in this scan, nothing to save.")
            return

        points_array = np.array(points_list, dtype=np.float32)

        # Numpy dosyasına kaydet
        np.save(self.output_path, points_array)

        self.get_logger().info(
            f"Saved first LiDAR scan: shape={points_array.shape}, path={self.output_path}"
        )

        # Bir daha kaydetmemek için flag’i kapat
        self.first_scan_saved = True


def main(args=None):
    rclpy.init(args=args)
    node = LidarDebug()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()