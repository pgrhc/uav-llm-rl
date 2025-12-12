import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import struct
from rclpy.qos import qos_profile_sensor_data


class RadarDebug(Node):
    def __init__(self):
        super().__init__("radar_debug")

        self.sub = self.create_subscription(
            PointCloud2,
            "/radar/points",
            self.cb,
            qos_profile_sensor_data
        )

        self.first_scan_saved = False
        self.output_path = "/home/ubuntu/Desktop/ros2_env/first_radar_scan.npy"

        self.get_logger().info("RadarDebug waiting for first radar scan...")

    def cb(self, msg):
        if self.first_scan_saved:
            return

        self.get_logger().info("Radar scan received. Parsing raw binary...")

        points = []

        step = msg.point_step      # bytes per point
        data = msg.data            # raw bytes
        fields = [(f.name, f.offset) for f in msg.fields]

        # x,y,z offset’lerini bul
        x_off = next(offset for name, offset in fields if name == "x")
        y_off = next(offset for name, offset in fields if name == "y")
        z_off = next(offset for name, offset in fields if name == "z")

        for i in range(0, len(data), step):
            try:
                x = struct.unpack_from('f', data, i + x_off)[0]
                y = struct.unpack_from('f', data, i + y_off)[0]
                z = struct.unpack_from('f', data, i + z_off)[0]

                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    points.append([x, y, z])
            except Exception:
                continue

        if len(points) == 0:
            self.get_logger().warn("No valid radar points found.")
        else:
            arr = np.array(points, dtype=np.float32)
            np.save(self.output_path, arr)
            self.get_logger().info(f"SAVED radar scan → shape={arr.shape}")

        self.first_scan_saved = True


def main(args=None):
    rclpy.init(args=args)
    node = RadarDebug()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()