import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from fusion_msgs.msg import FusionStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
from fusion_msgs.msg import RadarPoints

class SensorFusionSync(Node):
    def __init__(self):
        super().__init__("sensor_fusion_sync")

        # --- REAL ROS TOPICS (YOUR SYSTEM) ---
        self.cam_sub = Subscriber(
            self, Image,
            "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"
        )

        self.lidar_sub = Subscriber(
            self, PointCloud2,
            "/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan/points"
        )

        self.radar_sub = Subscriber(
            self, RadarPoints,
            "/radar/points_filtered_radarmsg"
        )

        self.odom_sub = Subscriber(
            self, Odometry,
            "/odometry/filtered"
        )

        # --- SYNCHRONIZER ---
        self.sync = ApproximateTimeSynchronizer(
            [self.cam_sub, self.lidar_sub, self.radar_sub, self.odom_sub],
            queue_size=50,
            slop=3.0,                   # sensör farklı hızlarda → toleransı artırdım
            allow_headerless=False
        )
        self.pub = self.create_publisher(FusionStamped, '/fusion/data', 10)
        self.sync.registerCallback(self.synced_callback)
        self.get_logger().info("Sensor Fusion Sync node started.")

    

    def synced_callback(self, cam_msg, lidar_msg, radar_msg, odom_msg):
        # self.get_logger().info(
        #     f"SYNC OK | "
        #     f"cam={cam_msg.header.stamp.sec}.{cam_msg.header.stamp.nanosec} | "
        #     f"lidar={lidar_msg.header.stamp.sec}.{lidar_msg.header.stamp.nanosec} | "
        #     f"radar={radar_msg.header.stamp.sec}.{radar_msg.header.stamp.nanosec} | "
        #     f"odom={odom_msg.header.stamp.sec}.{odom_msg.header.stamp.nanosec}"
        # )
        msg = FusionStamped()
        msg.stamp = cam_msg.header.stamp
        msg.image = cam_msg
        msg.lidar = lidar_msg
        msg.radar = radar_msg
        msg.odom = odom_msg
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionSync()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()