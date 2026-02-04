#!/usr/bin/env python3
"""
Route Safety Filter Node (Faz 0 - Basit versiyon)

RL/heuristic planner'dan gelen waypoint'i costmap'e göre kontrol eder.
Engele çok yakınsa, waypoint'i güvenli mesafeye çeker.

Faz 2'de CBF/RCBF + QP ile değiştirilecek.

Input:  /route/waypoint_desired (geometry_msgs/PoseStamped)
        /local_costmap/costmap (nav_msgs/OccupancyGrid)
        /odometry/filtered (nav_msgs/Odometry)
        /threat/state_vec (std_msgs/Float32MultiArray) - dinamik engeller için
Output: /route/waypoint_safe (geometry_msgs/PoseStamped)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

import numpy as np
import math


class RouteSafetyFilterNode(Node):
    def __init__(self):
        super().__init__('route_safety_filter_node')

        # Safety parameters
        self.safety_radius = 0.5  # meters - minimum distance from obstacles
        self.lethal_threshold = 90  # costmap value considered lethal (0-100)
        self.check_radius_cells = 10  # cells around waypoint to check

        # State
        self.latest_costmap = None
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0

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

        # Subscribers
        self.waypoint_sub = self.create_subscription(
            PoseStamped,
            '/route/waypoint_desired',
            self.waypoint_callback,
            10
        )

        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/local_costmap/costmap',
            self.costmap_callback,
            qos_reliable
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            qos_sensor
        )

        self.threat_sub = self.create_subscription(
            Float32MultiArray,
            '/threat/state_vec',
            self.threat_callback,
            10
        )

        # Publisher
        self.safe_waypoint_pub = self.create_publisher(
            PoseStamped,
            '/route/waypoint_safe',
            10
        )

        self.get_logger().info(
            f'RouteSafetyFilterNode started. Safety radius: {self.safety_radius}m'
        )

    def costmap_callback(self, msg: OccupancyGrid):
        """Store latest costmap."""
        self.latest_costmap = msg

    def odom_callback(self, msg: Odometry):
        """Update drone position."""
        self.drone_x = msg.pose.pose.position.x
        self.drone_y = msg.pose.pose.position.y
        self.drone_z = msg.pose.pose.position.z

    def threat_callback(self, msg: Float32MultiArray):
        """Store latest threat vector for dynamic obstacle avoidance."""
        self.latest_threat_vec = msg.data

    def waypoint_callback(self, msg: PoseStamped):
        """
        Receive desired waypoint, check safety, and publish safe waypoint.
        """
        if self.latest_costmap is None:
            # No costmap yet, pass through (unsafe but allows startup)
            self.get_logger().warn(
                'No costmap received yet, passing waypoint through',
                throttle_duration_sec=2.0
            )
            self.safe_waypoint_pub.publish(msg)
            return

        # Extract desired waypoint position
        wp_x = msg.pose.position.x
        wp_y = msg.pose.position.y
        wp_z = msg.pose.position.z

        # Check if waypoint is safe
        is_safe, nearest_obstacle_dist, obstacle_dir = self.check_waypoint_safety(wp_x, wp_y)

        if is_safe:
            # Waypoint is safe, pass through
            safe_msg = msg
        else:
            # Waypoint is unsafe, adjust it
            safe_msg = self.adjust_waypoint(msg, nearest_obstacle_dist, obstacle_dir)
            self.get_logger().info(
                f'Waypoint adjusted for safety. Obstacle dist: {nearest_obstacle_dist:.2f}m',
                throttle_duration_sec=0.5
            )

        # Publish safe waypoint
        safe_msg.header.stamp = self.get_clock().now().to_msg()
        self.safe_waypoint_pub.publish(safe_msg)

    def check_waypoint_safety(self, wp_x: float, wp_y: float):
        """
        Check if waypoint is safe (far enough from obstacles).
        
        Returns:
            is_safe: bool
            nearest_obstacle_dist: float (distance to nearest obstacle)
            obstacle_dir: tuple (dx, dy) direction to nearest obstacle (normalized)
        """
        costmap = self.latest_costmap
        resolution = costmap.info.resolution
        origin_x = costmap.info.origin.position.x
        origin_y = costmap.info.origin.position.y
        width = costmap.info.width
        height = costmap.info.height

        # Convert waypoint to costmap coordinates
        wp_px = int((wp_x - origin_x) / resolution)
        wp_py = int((wp_y - origin_y) / resolution)

        # Check if within costmap bounds
        if not (0 <= wp_px < width and 0 <= wp_py < height):
            # Outside costmap, consider safe but log warning
            self.get_logger().warn(
                'Waypoint outside costmap bounds',
                throttle_duration_sec=2.0
            )
            return True, float('inf'), (0.0, 0.0)

        # Convert costmap data to numpy
        data = np.array(costmap.data, dtype=np.int8).reshape((height, width))

        # Search for obstacles around waypoint
        nearest_dist = float('inf')
        obstacle_dx = 0.0
        obstacle_dy = 0.0

        # Safety radius in cells
        safety_cells = int(self.safety_radius / resolution)

        for dy in range(-self.check_radius_cells, self.check_radius_cells + 1):
            for dx in range(-self.check_radius_cells, self.check_radius_cells + 1):
                check_x = wp_px + dx
                check_y = wp_py + dy

                # Bounds check
                if not (0 <= check_x < width and 0 <= check_y < height):
                    continue

                # Check if cell is obstacle
                cell_value = data[check_y, check_x]
                if cell_value >= self.lethal_threshold:
                    # Calculate distance to this obstacle cell
                    dist = math.sqrt(dx * dx + dy * dy) * resolution

                    if dist < nearest_dist:
                        nearest_dist = dist
                        # Direction FROM obstacle TO waypoint
                        if dist > 0.01:
                            obstacle_dx = -dx * resolution / dist
                            obstacle_dy = -dy * resolution / dist

        # Check if within safety radius
        is_safe = nearest_dist > self.safety_radius

        return is_safe, nearest_dist, (obstacle_dx, obstacle_dy)

    def adjust_waypoint(self, msg: PoseStamped, obstacle_dist: float,
                        obstacle_dir: tuple) -> PoseStamped:
        """
        Adjust waypoint to maintain safe distance from obstacle.
        
        Strategy: Move waypoint away from obstacle along the obstacle direction,
        while trying to maintain progress towards the goal.
        """
        # How much we need to push away
        push_distance = self.safety_radius - obstacle_dist + 0.1  # +0.1 margin

        # Original waypoint
        wp_x = msg.pose.position.x
        wp_y = msg.pose.position.y

        # Direction from drone to waypoint
        dx_goal = wp_x - self.drone_x
        dy_goal = wp_y - self.drone_y
        dist_to_wp = math.sqrt(dx_goal * dx_goal + dy_goal * dy_goal)

        if dist_to_wp < 0.01:
            dist_to_wp = 0.01

        # Normalize
        dx_goal /= dist_to_wp
        dy_goal /= dist_to_wp

        # Push waypoint away from obstacle
        new_x = wp_x + obstacle_dir[0] * push_distance
        new_y = wp_y + obstacle_dir[1] * push_distance

        # Also reduce step size if needed (don't overshoot safe zone)
        new_dx = new_x - self.drone_x
        new_dy = new_y - self.drone_y
        new_dist = math.sqrt(new_dx * new_dx + new_dy * new_dy)

        # Limit maximum step to original distance
        if new_dist > dist_to_wp:
            scale = dist_to_wp / new_dist
            new_x = self.drone_x + new_dx * scale
            new_y = self.drone_y + new_dy * scale

        # Create adjusted message
        adjusted = PoseStamped()
        adjusted.header = msg.header
        adjusted.pose.position.x = new_x
        adjusted.pose.position.y = new_y
        adjusted.pose.position.z = msg.pose.position.z
        adjusted.pose.orientation = msg.pose.orientation

        return adjusted


def main(args=None):
    rclpy.init(args=args)
    node = RouteSafetyFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
