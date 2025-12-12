import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class CloudSaver(Node):
    def __init__(self):
        super().__init__("cloud_saver")

        topic = "/radar/points_filtered"   # ← buraya lidar/camera/radar istediğini yaz

        self.sub = self.create_subscription(
            PointCloud2,
            topic,
            self.cb,
            10
        )

        self.saved = False
        self.output_path = "/home/ubuntu/Desktop/radar_first_frame.npy"

        self.get_logger().info(f"Listening for first frame on {topic}")

    def cb(self, msg):
        if self.saved:
            return

        points = []
        for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])

        arr = np.array(points, dtype=np.float32)
        np.save(self.output_path, arr)

        self.get_logger().info(f"Saved frame → shape={arr.shape}")
        self.saved = True


def main(args=None):
    rclpy.init(args=args)
    node = CloudSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()