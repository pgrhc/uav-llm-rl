import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import struct
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from fusion_msgs.msg import RadarPoints, RadarPoint




class RadarFilter(Node):
    def __init__(self):
        super().__init__("radar_filter")

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Publisher → RadarPoints
        self.pub = self.create_publisher(RadarPoints, "/radar/points_filtered_radarmsg", qos)

        # Subscriber → raw radar points (PointCloud2)
        self.sub = self.create_subscription(
            PointCloud2,
            "/radar/points",
            self.cb,
            qos
        )

        self.get_logger().info("RadarFilter running with RadarPoints msg...")

    def cb(self, msg):
        # Convert PointCloud2 to numpy array
        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])

        if len(pts) == 0:
            return

        pts = np.array(pts)
        pts = pts[np.isfinite(pts).all(axis=1)]

        if len(pts) == 0:
            return

        pts = pts[pts[:, 2] > -0.25] 

        if len(pts) == 0:
            return

        # ---------- 1) Gaussian Beam Pattern ----------
        angles = np.arctan2(pts[:,1], pts[:,0])
        sigma = np.radians(20)
        weight = np.exp(-(angles**2) / (2*sigma*sigma))
        mask = np.random.rand(len(pts)) < weight
        pts = pts[mask]

        # ---------- 2) Range Thinning ----------
        if len(pts) > 200:
            idx = np.linspace(0, len(pts)-1, 200).astype(int)
            pts = pts[idx]

        # ---------- 3) Angle Resolution reduction ----------
        angle_bins = np.linspace(-np.pi/2, np.pi/2, 64)
        bin_idx = np.digitize(np.arctan2(pts[:,1], pts[:,0]), angle_bins)
        _, unique_idx = np.unique(bin_idx, return_index=True)
        pts = pts[unique_idx]

        # ---------- 4) Random Drops ----------
        drop_mask = np.random.rand(len(pts)) > 0.1
        pts = pts[drop_mask]

        # ---------- 5) Radar-like noise ----------
        range_noise = np.random.normal(0, 0.05, len(pts))
        angle_noise = np.random.normal(0, np.radians(1), len(pts))

        r = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
        theta = np.arctan2(pts[:,1], pts[:,0])

        r = np.abs(r + range_noise)
        theta = theta + angle_noise

        pts[:,0] = r * np.cos(theta)
        pts[:,1] = r * np.sin(theta)

        # ---------- 6) Intensity ----------
        intensity = np.exp(-r / 15.0)

        # ---------- 7) Compute RANGE / AZIMUTH / ELEVATION ----------
        rng = np.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
        azim = np.arctan2(pts[:,1], pts[:,0])
        elev = np.arctan2(pts[:,2], np.sqrt(pts[:,0]**2 + pts[:,1]**2))

        # ---------- Build RadarPoints message ----------
        out = RadarPoints()
        out.header = msg.header
        out.header.frame_id = "radar_link"

        for i, p in enumerate(pts):
            rp = RadarPoint()
            rp.x = float(p[0])
            rp.y = float(p[1])
            rp.z = float(p[2])
            rp.intensity = float(intensity[i])
            rp.range = float(rng[i])
            rp.azimuth = float(azim[i])
            rp.elevation = float(elev[i])

            out.points.append(rp)

        # Publish
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = RadarFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()