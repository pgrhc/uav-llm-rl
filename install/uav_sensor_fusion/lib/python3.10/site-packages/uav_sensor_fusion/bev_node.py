import rclpy
from rclpy.node import Node
import numpy as np

from fusion_msgs.msg import FusionStamped
from nav_msgs.msg import OccupancyGrid

import sensor_msgs_py.point_cloud2 as pc2

# TF packages
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped


class BEVNode(Node):
    def __init__(self):
        super().__init__("bev_node")

        # Listen for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscriber: fusion data
        self.sub = self.create_subscription(
            FusionStamped,
            "/fusion/data",
            self.fusion_callback,
            10
        )

        # Publisher: BEV occupancy grid
        self.pub = self.create_publisher(OccupancyGrid, "/bev/map", 10)

        # BEV parameters
        self.map_size = 100.0        # meters
        self.resolution = 0.2        # meters per cell
        self.grid_size = int(self.map_size / self.resolution)

        self.get_logger().info("BEV node with TF started.")


    def fusion_callback(self, msg):

        # Empty BEV grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Process LiDAR
        lidar_points = pc2.read_points_list(msg.lidar,
                                            field_names=("x","y","z"),
                                            skip_nans=True)
        self.add_pointcloud_to_grid(lidar_points, msg.lidar.header.frame_id, grid)

        # Process Radar
        radar_points = pc2.read_points_list(msg.radar,
                                            field_names=("x","y","z"),
                                            skip_nans=True)
        self.add_pointcloud_to_grid(radar_points, msg.radar.header.frame_id, grid)

        # Publish grid
        bev_msg = self.build_grid_msg(grid, msg.stamp)
        self.pub.publish(bev_msg)


    def add_pointcloud_to_grid(self, points, frame_id, grid):

        # Try to lookup transform (sensor_frame -> odom)
        try:
            transform = self.tf_buffer.lookup_transform(
                "odom",
                frame_id,
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"No TF from {frame_id} to odom yet.")
            return

        for p in points:

            # skip invalid points
            if not np.isfinite(p[0]) or not np.isfinite(p[1]):
                continue

            # build point for TF conversion
            pt = PointStamped()
            pt.header.frame_id = frame_id
            pt.point.x = float(p[0])
            pt.point.y = float(p[1])
            pt.point.z = float(p[2])

            # transform point to odom frame
            try:
                pt_global = tf2_geometry_msgs.do_transform_point(pt, transform)
            except Exception:
                continue

            x = pt_global.point.x
            y = pt_global.point.y

            # convert to BEV grid coordinates
            gx = int(x / self.resolution + self.grid_size / 2)
            gy = int(y / self.resolution + self.grid_size / 2)

            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                grid[gy, gx] = 100


    def build_grid_msg(self, grid, stamp):
        msg = OccupancyGrid()

        msg.header.frame_id = "odom"  # RViz için en güvenli FRAME
        msg.header.stamp = stamp

        msg.info.resolution = self.resolution
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size

        msg.info.origin.position.x = - (self.map_size / 2)
        msg.info.origin.position.y = - (self.map_size / 2)
        msg.info.origin.orientation.w = 1.0

        msg.data = grid.flatten().tolist()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = BEVNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()