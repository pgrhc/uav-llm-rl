#!/usr/bin/env python3
"""
Route Safety Filter Node (Faz 0 - Basit versiyon)

RL/heuristic planner'dan gelen waypoint'i costmap'e göre kontrol eder.
Engele çok yakınsa, waypoint'i güvenli mesafeye çeker.

Faz 2'de CBF/RCBF + QP ile değiştirilecek.

Input:  /route/waypoint_desired (geometry_msgs/PoseStamped)
        /local_costmap/costmap (nav_msgs/OccupancyGrid)
        /odometry/filtered (nav_msgs/Odometry)
        /threat/state_vec (std_msgs/Float32MultiArray) — ThreatEncoderV2 düzeni:
        [3 ego][36 lidar][K=5 × 7 token]; token = class,r,closing,sn,cs,conf,vis
        (sn,cs) birim vektör base_link’te tehdit yönü; r mesafe (m).
        Filtre: maliyet haritasına ek olarak bu vektörle yakın/hizalı tehditlerde
        effective safety radius şişirilir (parametrelerle kapatılabilir).
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
        self.declare_parameter('safety_radius', 0.5)
        self.declare_parameter('lethal_threshold', 90)
        self.declare_parameter('check_radius_cells', 10)
        self.declare_parameter('use_threat_vec', True)
        self.declare_parameter('threat_lidar_sectors', 36)
        self.declare_parameter('threat_k_tracks', 5)
        self.declare_parameter('threat_token_len', 7)
        self.declare_parameter('threat_conf_min', 0.15)
        self.declare_parameter('threat_max_inflate', 0.85)
        self.declare_parameter('threat_align_dot_threshold', 0.75)
        self.declare_parameter('threat_align_bonus', 0.35)

        self.safety_radius = float(self.get_parameter('safety_radius').value)
        self.lethal_threshold = int(self.get_parameter('lethal_threshold').value)
        self.check_radius_cells = int(self.get_parameter('check_radius_cells').value)
        self.use_threat_vec = bool(self.get_parameter('use_threat_vec').value)
        self.threat_lidar_sectors = int(self.get_parameter('threat_lidar_sectors').value)
        self.threat_k_tracks = int(self.get_parameter('threat_k_tracks').value)
        self.threat_token_len = int(self.get_parameter('threat_token_len').value)
        self.threat_conf_min = float(self.get_parameter('threat_conf_min').value)
        self.threat_max_inflate = float(self.get_parameter('threat_max_inflate').value)
        self.threat_align_dot_threshold = float(
            self.get_parameter('threat_align_dot_threshold').value
        )
        self.threat_align_bonus = float(self.get_parameter('threat_align_bonus').value)

        self._threat_token_offset = 3 + self.threat_lidar_sectors

        # State
        self.latest_costmap = None
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.drone_yaw = 0.0

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
        qos_costmap = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10,
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
            qos_costmap
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
            f'RouteSafetyFilterNode started. safety_radius={self.safety_radius}m, '
            f'use_threat_vec={self.use_threat_vec}, threat_token_offset={self._threat_token_offset}'
        )

    def costmap_callback(self, msg: OccupancyGrid):
        """Store latest costmap."""
        self.latest_costmap = msg

    def odom_callback(self, msg: Odometry):
        """Update drone position."""
        self.drone_x = msg.pose.pose.position.x
        self.drone_y = msg.pose.pose.position.y
        self.drone_z = msg.pose.pose.position.z
        qx = float(msg.pose.pose.orientation.x)
        qy = float(msg.pose.pose.orientation.y)
        qz = float(msg.pose.pose.orientation.z)
        qw = float(msg.pose.pose.orientation.w)
        # threat_encoder_v2 ile aynı yaw (base_link / odom düzlemi)
        self.drone_yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

    def threat_callback(self, msg: Float32MultiArray):
        """Store latest threat vector for dynamic obstacle avoidance."""
        self.latest_threat_vec = msg.data

    def _dynamic_threat_radius_extra(self, wp_x: float, wp_y: float) -> float:
        """
        ThreatEncoderV2 state_vec üzerinden base_link uyumlu ek güvenlik mesafesi.
        Harita maliyeti statik; dinamik tehdit için yarıçap şişirilir (hedefe doğru
        hizalı ve yakın tehditte ekstra).
        """
        if not self.use_threat_vec or self.latest_threat_vec is None:
            return 0.0
        data = self.latest_threat_vec
        if not data:
            return 0.0
        need = self._threat_token_offset + self.threat_k_tracks * self.threat_token_len
        if len(data) < need:
            return 0.0

        dx = wp_x - self.drone_x
        dy = wp_y - self.drone_y
        dwp = math.hypot(dx, dy)
        if dwp < 1e-6:
            return 0.0
        cy = math.cos(self.drone_yaw)
        sy = math.sin(self.drone_yaw)
        bx = cy * dx + sy * dy
        by = -sy * dx + cy * dy
        inv = 1.0 / dwp
        ux, uy = bx * inv, by * inv

        base = 0.0
        aligned = 0.0
        t0 = self._threat_token_offset
        L = self.threat_token_len
        for i in range(self.threat_k_tracks):
            off = t0 + i * L
            r = float(data[off + 1])
            closing = float(data[off + 2])
            sn = float(data[off + 3])
            cs = float(data[off + 4])
            conf = float(data[off + 5])
            vis = float(data[off + 6])
            if r < 0.15 or conf < self.threat_conf_min or vis < 0.25:
                continue
            proximity = max(0.0, min(1.0, (12.0 - r) / 12.0))
            closing_w = 1.0 + 0.35 * max(0.0, closing)
            contrib = 0.18 * conf * proximity * closing_w
            base += contrib
            dot = abs(ux * cs + uy * sn)
            if dot >= self.threat_align_dot_threshold and r < 10.0 and vis > 0.45:
                aligned = max(aligned, self.threat_align_bonus * conf * proximity)

        extra = min(self.threat_max_inflate, base + aligned)
        return float(extra)

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

        eff_radius = self.safety_radius + self._dynamic_threat_radius_extra(wp_x, wp_y)

        # Check if waypoint is safe
        is_safe, nearest_obstacle_dist, obstacle_dir = self.check_waypoint_safety(
            wp_x, wp_y, eff_radius
        )

        if is_safe:
            # Waypoint is safe, pass through
            safe_msg = msg
        else:
            # Waypoint is unsafe, adjust it
            safe_msg = self.adjust_waypoint(
                msg, nearest_obstacle_dist, obstacle_dir, eff_radius
            )
            self.get_logger().info(
                f'Waypoint adjusted for safety. Obstacle dist: {nearest_obstacle_dist:.2f}m',
                throttle_duration_sec=0.5
            )

        # Publish safe waypoint
        safe_msg.header.stamp = self.get_clock().now().to_msg()
        self.safe_waypoint_pub.publish(safe_msg)

    def check_waypoint_safety(self, wp_x: float, wp_y: float, safety_radius: float):
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

        # Check if within safety radius (costmap mesafesi; dinamik tehdit eff_radius ile)
        is_safe = nearest_dist > safety_radius

        return is_safe, nearest_dist, (obstacle_dx, obstacle_dy)

    def adjust_waypoint(self, msg: PoseStamped, obstacle_dist: float,
                        obstacle_dir: tuple, safety_radius: float) -> PoseStamped:
        """
        Adjust waypoint to maintain safe distance from obstacle.
        
        Strategy: Move waypoint away from obstacle along the obstacle direction,
        while trying to maintain progress towards the goal.
        """
        # How much we need to push away
        push_distance = safety_radius - obstacle_dist + 0.1  # +0.1 margin

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
