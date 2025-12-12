import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from px4_msgs.msg import VehicleOdometry
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion


class Px4OdomBridge(Node):
    def __init__(self):
        super().__init__("px4_odom_bridge")

        # PX4 yayıncısına uyumlu QoS profili
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.sub = self.create_subscription(
            VehicleOdometry,
            "/fmu/out/vehicle_odometry",
            self.cb,
            qos,
        )

        self.pub = self.create_publisher(
            Odometry,
            "/odom/px4",
            10,
        )

        self.get_logger().info("px4_odom_bridge started, waiting for /fmu/out/vehicle_odometry...")


    def cb(self, msg: VehicleOdometry):
        

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(msg.position[0])
        odom.pose.pose.position.y = float(msg.position[1])
        odom.pose.pose.position.z = float(msg.position[2])

        q = Quaternion()
        q.w = float(msg.q[0])
        q.x = float(msg.q[1])
        q.y = float(msg.q[2])
        q.z = float(msg.q[3])
        odom.pose.pose.orientation = q

        odom.twist.twist.linear.x = float(msg.velocity[0])
        odom.twist.twist.linear.y = float(msg.velocity[1])
        odom.twist.twist.linear.z = float(msg.velocity[2])

        self.pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = Px4OdomBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()