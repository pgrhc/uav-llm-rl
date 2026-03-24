#!/usr/bin/env python3
"""
Costmap Patch Node

Subscribes to Nav2 local costmap and extracts a drone-centered patch
for use as CNN input in the route planning agent.

Patch geometry: varsayılan olarak costmap grid eksenlerine hizalı (map/odom düzemiyle
aynı yön); gövde önü ile görüntü "yukarı" değildir. İsteğe bağlı align_patch_to_body
ile patch merkez etrafında -yaw döndürülür (OpenCV); frame_id o zaman base_link.

Input:  /local_costmap/costmap (nav_msgs/OccupancyGrid)
        /odometry/filtered (nav_msgs/Odometry)
Output: /route/costmap_patch (sensor_msgs/Image, 64x64 mono8)
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False


class CostmapPatchNode(Node):
    def __init__(self):
        super().__init__('costmap_patch_node')

        self.declare_parameter('patch_size', 64)
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('align_patch_to_body', False)
        self.declare_parameter('align_rotation_deg_offset', 0.0)
        self.declare_parameter('output_frame_id', '')

        self.patch_size = int(self.get_parameter('patch_size').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.align_patch_to_body = bool(self.get_parameter('align_patch_to_body').value)
        self.align_rotation_deg_offset = float(
            self.get_parameter('align_rotation_deg_offset').value
        )
        self.output_frame_id = str(self.get_parameter('output_frame_id').value).strip()

        if self.align_patch_to_body and not _HAS_CV2:
            self.get_logger().error(
                'align_patch_to_body requires cv2 (OpenCV); disabling rotation.'
            )
            self.align_patch_to_body = False

        # State
        self.latest_costmap = None
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_yaw = 0.0
        self.bridge = CvBridge()

        # QoS profiles
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Subscribers
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

        # Publisher
        self.patch_pub = self.create_publisher(
            Image,
            '/route/costmap_patch',
            10
        )

        # Timer for publishing at fixed rate
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_patch)

        self.get_logger().info(
            f'CostmapPatchNode started. patch={self.patch_size}x{self.patch_size}, '
            f'align_patch_to_body={self.align_patch_to_body}'
        )

    def costmap_callback(self, msg: OccupancyGrid):
        """Store latest costmap."""
        self.latest_costmap = msg

    def odom_callback(self, msg: Odometry):
        """Update drone position."""
        self.drone_x = msg.pose.pose.position.x
        self.drone_y = msg.pose.pose.position.y
        qx = float(msg.pose.pose.orientation.x)
        qy = float(msg.pose.pose.orientation.y)
        qz = float(msg.pose.pose.orientation.z)
        qw = float(msg.pose.pose.orientation.w)
        self.drone_yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

    def publish_patch(self):
        """Extract drone-centered patch and publish as Image."""
        if self.latest_costmap is None:
            return

        costmap = self.latest_costmap

        # Costmap parameters
        width = costmap.info.width
        height = costmap.info.height
        resolution = costmap.info.resolution
        origin_x = costmap.info.origin.position.x
        origin_y = costmap.info.origin.position.y

        # Convert costmap data to numpy array
        # OccupancyGrid values: -1 (unknown), 0-100 (free-occupied)
        data = np.array(costmap.data, dtype=np.int8).reshape((height, width))

        # Convert to uint8: -1 -> 128 (unknown), 0 -> 0 (free), 100 -> 255 (occupied)
        patch_data = np.zeros((height, width), dtype=np.uint8)
        patch_data[data == -1] = 128  # Unknown
        patch_data[data >= 0] = (data[data >= 0] * 2.55).astype(np.uint8)  # 0-100 -> 0-255

        # Find drone position in costmap coordinates (pixel indices)
        drone_px = int((self.drone_x - origin_x) / resolution)
        drone_py = int((self.drone_y - origin_y) / resolution)

        # Extract patch centered on drone
        half = self.patch_size // 2

        # Calculate crop bounds with boundary handling
        x_start = drone_px - half
        x_end = drone_px + half
        y_start = drone_py - half
        y_end = drone_py + half

        # Create output patch (initialized with unknown=128)
        patch = np.full((self.patch_size, self.patch_size), 128, dtype=np.uint8)

        # Calculate valid regions
        src_x_start = max(0, x_start)
        src_x_end = min(width, x_end)
        src_y_start = max(0, y_start)
        src_y_end = min(height, y_end)

        dst_x_start = src_x_start - x_start
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_start = src_y_start - y_start
        dst_y_end = dst_y_start + (src_y_end - src_y_start)

        # Copy valid region
        if src_x_end > src_x_start and src_y_end > src_y_start:
            patch[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                patch_data[src_y_start:src_y_end, src_x_start:src_x_end]

        if self.align_patch_to_body and _HAS_CV2:
            h, w = patch.shape
            center = (w / 2.0, h / 2.0)
            angle_deg = -math.degrees(self.drone_yaw) + self.align_rotation_deg_offset
            rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            patch = cv2.warpAffine(
                patch, rot, (w, h), flags=cv2.INTER_NEAREST, borderValue=128
            )

        if self.output_frame_id:
            frame_id = self.output_frame_id
        elif self.align_patch_to_body:
            frame_id = 'base_link'
        else:
            frame_id = costmap.header.frame_id or 'map'

        # Publish as Image
        img_msg = self.bridge.cv2_to_imgmsg(patch, encoding='mono8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = frame_id

        self.patch_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CostmapPatchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
