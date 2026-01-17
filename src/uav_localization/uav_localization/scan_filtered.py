import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from px4_msgs.msg import VehicleOdometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math

class ScanFilter(Node):
    def __init__(self):
        super().__init__('scan_filter_node')

        # CONFIGURATION
        self.MAX_TILT_ANGLE = 0.05  # Radians (~3 degrees). Tilted more? Block the scan.
        self.current_pitch = 0.0

        # 1. Listen to PX4 Odometry to get the Drone's Angle
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_profile
        )

        # 2. Listen to the "Dirty" Scan (Your long Gazebo topic)
        # PASTE YOUR LONG TOPIC HERE
        self.input_scan_topic = '/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan'
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.input_scan_topic,
            self.scan_callback,
            10
        )

        # 3. Publish the "Clean" Scan
        self.scan_pub = self.create_publisher(LaserScan, '/scan_filtered', 10)
        
        self.get_logger().info("Scan Filter Running... preventing ground hits!")

    def odom_callback(self, msg):
        # Convert Quaternion to Euler (Pitch only)
        # PX4 uses [w, x, y, z]
        q0, q1, q2, q3 = msg.q[0], msg.q[1], msg.q[2], msg.q[3]
        
        # Math to get Pitch (Tilt)
        sin_p = 2 * (q0 * q2 - q3 * q1)
        if abs(sin_p) >= 1:
            self.current_pitch = math.copysign(math.pi / 2, sin_p)
        else:
            self.current_pitch = math.asin(sin_p)

    def scan_callback(self, msg):
        # The Gatekeeper Logic
        if abs(self.current_pitch) < self.MAX_TILT_ANGLE:
            # Drone is flat -> Publish the scan!
            self.scan_pub.publish(msg)
        else:
            # Drone is tilted -> Do nothing (Drop the packet)
            # This protects the map from seeing the floor.
            pass

def main(args=None):
    rclpy.init(args=args)
    node = ScanFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()