#!/usr/bin/env python3
"""
Heuristic Planner Node (Faz 0 - RL olmadan test için)

Hedefe doğru sabit adımlı waypoint üretir.
Sonradan RL policy ile değiştirilecek.

Input:  /route/costmap_patch (sensor_msgs/Image) - şimdilik kullanılmıyor
        /odometry/filtered (nav_msgs/Odometry)
        /goal_pose (geometry_msgs/PoseStamped)
        /threat/state_vec (std_msgs/Float32MultiArray) - şimdilik kullanılmıyor
Output: /route/waypoint_desired (geometry_msgs/PoseStamped)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

import numpy as np
import math


class HeuristicPlannerNode(Node):
    def __init__(self):
        super().__init__('heuristic_planner_node')

        # Parameters
        self.step_size = 0.3  # meters per waypoint step
        self.planning_rate = 5.0  # Hz (2-5 Hz as specified)
        self.goal_tolerance = 0.5  # meters

        # State
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.drone_yaw = 0.0

        self.goal_x = None
        self.goal_y = None
        self.goal_z = None

        self.latest_costmap_patch = None
        self.latest_threat_vec = None

        # QoS profiles
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        qos_transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            qos_sensor
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            qos_transient
        )

        self.costmap_patch_sub = self.create_subscription(
            Image,
            '/route/costmap_patch',
            self.costmap_patch_callback,
            10
        )

        self.threat_sub = self.create_subscription(
            Float32MultiArray,
            '/threat/state_vec',
            self.threat_callback,
            10
        )

        # Publisher
        self.waypoint_pub = self.create_publisher(
            PoseStamped,
            '/route/waypoint_desired',
            10
        )

        # Timer for planning
        self.timer = self.create_timer(1.0 / self.planning_rate, self.plan_waypoint)

        self.get_logger().info(
            f'HeuristicPlannerNode started. Step: {self.step_size}m, Rate: {self.planning_rate}Hz'
        )

    def odom_callback(self, msg: Odometry):
        """Update drone state."""
        self.drone_x = msg.pose.pose.position.x
        self.drone_y = msg.pose.pose.position.y
        self.drone_z = msg.pose.pose.position.z

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.drone_yaw = math.atan2(siny_cosp, cosy_cosp)

    def goal_callback(self, msg: PoseStamped):
        """Receive goal pose."""
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.goal_z = msg.pose.position.z
        self.get_logger().info(
            f'Goal received: ({self.goal_x:.2f}, {self.goal_y:.2f}, {self.goal_z:.2f})'
        )

    def costmap_patch_callback(self, msg: Image):
        """Store latest costmap patch (for future RL use)."""
        self.latest_costmap_patch = msg

    def threat_callback(self, msg: Float32MultiArray):
        """Store latest threat vector (for future RL use)."""
        self.latest_threat_vec = msg.data

    def plan_waypoint(self):
        """Generate next waypoint towards goal."""
        # Check if we have a goal
        if self.goal_x is None or self.goal_y is None:
            return

        # Calculate distance to goal
        dx = self.goal_x - self.drone_x
        dy = self.goal_y - self.drone_y
        dist_2d = math.sqrt(dx * dx + dy * dy)

        # Check if already at goal
        if dist_2d < self.goal_tolerance:
            self.get_logger().info('Goal reached!', throttle_duration_sec=2.0)
            return

        # Normalize direction and apply step size
        if dist_2d > self.step_size:
            dx = dx / dist_2d * self.step_size
            dy = dy / dist_2d * self.step_size

        # Calculate desired yaw (face towards goal)
        desired_yaw = math.atan2(dy, dx)

        # Create waypoint message (in odom frame)
        waypoint = PoseStamped()
        waypoint.header.stamp = self.get_clock().now().to_msg()
        waypoint.header.frame_id = 'odom'

        # Absolute position (current + delta)
        waypoint.pose.position.x = self.drone_x + dx
        waypoint.pose.position.y = self.drone_y + dy
        waypoint.pose.position.z = self.drone_z  # Keep current altitude

        # Orientation (yaw only)
        waypoint.pose.orientation.z = math.sin(desired_yaw / 2.0)
        waypoint.pose.orientation.w = math.cos(desired_yaw / 2.0)

        # Publish
        self.waypoint_pub.publish(waypoint)


def main(args=None):
    rclpy.init(args=args)
    node = HeuristicPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
