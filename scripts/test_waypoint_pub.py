#!/usr/bin/env python3
"""
Waypoint Test Script — Drone hareketini izole test etmek için

Kullanım:
  1. follow_path ve route_goal_navigator çalışıyor olmalı
  2. Bu scripti çalıştır: python scripts/test_waypoint_pub.py
  3. Drone hareket ediyorsa → sorun train/env tarafında
  4. Drone hareket etmiyorsa → sorun follow_path veya PX4 tarafında

Script mevcut drone pozisyonundan 2m ileriye waypoint yayınlar.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import math
import time


class WaypointTestNode(Node):
    def __init__(self):
        super().__init__("waypoint_test_node")
        self.wp_pub = self.create_publisher(
            PoseStamped, "/route/waypoint_desired", 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odometry/filtered", self._cb_odom, 10
        )
        self.pos = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.odom_ok = False
        self.timer = self.create_timer(0.2, self._publish_waypoint)
        self.get_logger().info("Waypoint test başladı — 2m ileri waypoint yayınlanıyor")

    def _cb_odom(self, msg):
        self.pos[0] = msg.pose.pose.position.x
        self.pos[1] = msg.pose.pose.position.y
        self.pos[2] = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)
        self.odom_ok = True

    def _publish_waypoint(self):
        if not self.odom_ok:
            return
        # 2m ileri (yaw yönünde)
        dx = 2.0 * math.cos(self.yaw)
        dy = 2.0 * math.sin(self.yaw)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.pose.position.x = self.pos[0] + dx
        msg.pose.position.y = self.pos[1] + dy
        msg.pose.position.z = self.pos[2]
        msg.pose.orientation.z = math.sin(self.yaw / 2.0)
        msg.pose.orientation.w = math.cos(self.yaw / 2.0)
        self.wp_pub.publish(msg)


def main():
    rclpy.init()
    node = WaypointTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
