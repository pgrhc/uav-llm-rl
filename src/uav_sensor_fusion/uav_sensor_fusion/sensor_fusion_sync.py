import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from fusion_msgs.msg import FusionStamped, RadarPoints

class SensorFusionSync(Node):
    def __init__(self):
        super().__init__("sensor_fusion_sync")

        self.last_cam = None
        self.last_lidar = None
        self.last_radar = None
        self.last_odom = None

        # Store message stamps (builtin_interfaces/Time)
        self.st_cam = None
        self.st_lidar = None
        self.st_radar = None
        self.st_odom = None

        self.timeout_sec = 1.0

        self.create_subscription(
            Image,
            "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image",
            self.cb_cam, 10
        )
        self.create_subscription(
            PointCloud2,
            "/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan/points",
            self.cb_lidar, 10
        )
        self.create_subscription(
            RadarPoints,
            "/radar/points_filtered_radarmsg",
            self.cb_radar, 10
        )
        self.create_subscription(
            Odometry,
            "/odometry/filtered",
            self.cb_odom, 10
        )

        self.pub = self.create_publisher(FusionStamped, "/fusion/data", 10)
        self.timer = self.create_timer(0.1, self.publish_fused)

        self.get_logger().info("Sensor Fusion Sync (robust) node started.")

    def now_ns(self):
        return self.get_clock().now().nanoseconds

    def stamp_to_ns(self, stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def is_fresh_stamp(self, stamp):
        if stamp is None:
            return False
        age_ns = self.now_ns() - self.stamp_to_ns(stamp)
        return (age_ns / 1e9) <= self.timeout_sec

    def cb_cam(self, msg):
        self.last_cam = msg
        self.st_cam = msg.header.stamp

    def cb_lidar(self, msg):
        self.last_lidar = msg
        self.st_lidar = msg.header.stamp

    def cb_radar(self, msg):
        # If you want “no obstacle” to still publish: radar node should publish empty points.
        self.last_radar = msg
        self.st_radar = msg.header.stamp

    def cb_odom(self, msg):
        self.last_odom = msg
        self.st_odom = msg.header.stamp

    def make_empty_lidar(self, ref_stamp, ref_frame):
        pc = PointCloud2()
        pc.header.stamp = ref_stamp
        pc.header.frame_id = ref_frame
        # width/height/data left as default => empty cloud
        return pc

    def publish_fused(self):
        # Require at least cam + odom, and both fresh
        if self.last_cam is None or self.last_odom is None:
            return
        if not self.is_fresh_stamp(self.st_cam) or not self.is_fresh_stamp(self.st_odom):
            return

        msg = FusionStamped()
        msg.stamp = self.last_cam.header.stamp
        msg.image = self.last_cam
        msg.odom = self.last_odom

        # Lidar
        if self.last_lidar is not None and self.is_fresh_stamp(self.st_lidar):
            msg.lidar = self.last_lidar
        else:
            msg.lidar = self.make_empty_lidar(msg.stamp, self.last_cam.header.frame_id)

        # Radar
        if self.last_radar is not None and self.is_fresh_stamp(self.st_radar):
            msg.radar = self.last_radar
        else:
            empty_r = RadarPoints()
            # if RadarPoints has header, set it; if not, ignore
            if hasattr(empty_r, "header"):
                empty_r.header.stamp = msg.stamp
            msg.radar = empty_r

        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionSync()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()