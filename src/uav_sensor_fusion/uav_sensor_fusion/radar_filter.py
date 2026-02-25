import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from fusion_msgs.msg import RadarPoints, RadarPoint
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration
import tf2_py

class RadarFilter(Node):
    def __init__(self):
        super().__init__("radar_filter")
        
        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Platform tespit parametreleri
        self.declare_parameter('aerial_threshold', 1.0)  # 1m üstü = havada
        self.declare_parameter('reference_frame', 'odom')  # veya 'map'
        self.declare_parameter('radar_frame', 'radar_link')
        
        self.aerial_threshold = self.get_parameter('aerial_threshold').value
        self.reference_frame = self.get_parameter('reference_frame').value
        self.radar_frame = self.get_parameter('radar_frame').value
        
        # IWR6348 parametreleri
        self.declare_parameter('azimuth_fov', 120.0)
        self.declare_parameter('elevation_fov', 30.0)
        self.declare_parameter('azimuth_resolution', 15.0)
        self.declare_parameter('max_range', 80.0)
        self.declare_parameter('min_range', 0.5)
        self.declare_parameter('max_points', 128)
        
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        
        self.pub = self.create_publisher(RadarPoints, "/radar/points_filtered_radarmsg", qos)
        self.sub = self.create_subscription(
            PointCloud2,
            "/radar/points",
            self.cb,
            qos
        )
        
        # Durum değişkenleri
        self.current_height = 0.0
        self.is_aerial = False
        self.platform_type = "unknown"
        
        self.get_logger().info("IWR6348 Radar Filter running with AUTO platform detection...")

    def get_radar_height(self):
        """Radarın zemine olan yüksekliğini TF'den al"""
        try:
            # Radar'dan referans frame'e transform al
            transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                self.radar_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1)
            )
            
            # Z koordinatı = yükseklik
            height = transform.transform.translation.z
            
            # Platform tipini güncelle
            if height > self.aerial_threshold:
                if not self.is_aerial:
                    self.get_logger().info(f"🚁 AERIAL mode detected (height: {height:.2f}m)")
                self.is_aerial = True
                self.platform_type = "aerial"
            else:
                if self.is_aerial:
                    self.get_logger().info(f"🚗 GROUND mode detected (height: {height:.2f}m)")
                self.is_aerial = False
                self.platform_type = "ground"
            
            self.current_height = height
            return height
            
        except (tf2_py.LookupException, tf2_py.ConnectivityException, 
                tf2_py.ExtrapolationException) as e:
            # TF henüz hazır değil veya hata var
            if self.platform_type == "unknown":
                self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=5.0)
            return self.current_height

    def cb(self, msg):
        # Her callback'te yüksekliği güncelle
        height = self.get_radar_height()
        
        # PointCloud2 → numpy
        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        
        if len(pts) == 0:
            return
        
        pts = np.array(pts)
        pts = pts[np.isfinite(pts).all(axis=1)]
        
        if len(pts) == 0:
            return
        
        # ---------- IWR6348 Filtreler ----------
        
        # 1) Range limitleri
        r = np.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
        mask_range = (r >= 0.5) & (r <= 80.0)
        pts = pts[mask_range]
        r = r[mask_range]
        
        if len(pts) == 0:
            return
        
        # 2) FOV filtreleme
        azim = np.arctan2(pts[:,1], pts[:,0])
        elev = np.arctan2(pts[:,2], np.sqrt(pts[:,0]**2 + pts[:,1]**2))
        
        azim_fov = np.radians(60)
        elev_fov = np.radians(15)
        
        mask_fov = (np.abs(azim) <= azim_fov) & (np.abs(elev) <= elev_fov)
        pts = pts[mask_fov]
        r = r[mask_fov]
        azim = azim[mask_fov]
        elev = elev[mask_fov]
        
        if len(pts) == 0:
            return
        
        # 3) OTOMATIK Zemin/Tavan filtreleme
        if self.is_aerial:
            # Havada: Geniş Z limiti (hem yukarı hem aşağı bakabilir)
            # Sadece radarın kendisine çok yakın noktaları at
            z_min = -height - 5.0  # Zeminden 5m aşağısı
            z_max = 50.0  # 50m yukarısı
            mask_z = (pts[:, 2] > z_min) & (pts[:, 2] < z_max)
            
        else:
            # Yerde: Zemin altını filtrele
            # Radarın mounting height'ını hesaba kat
            ground_threshold = -height - 0.3
            mask_z = pts[:, 2] > ground_threshold
        
        pts = pts[mask_z]
        r = r[mask_z]
        azim = azim[mask_z]
        elev = elev[mask_z]
        
        if len(pts) == 0:
            return
        
        # 4) Gaussian Antenna Pattern
        azim_sigma = np.radians(20)
        elev_sigma = np.radians(10)
        
        weight_azim = np.exp(-(azim**2) / (2*azim_sigma**2))
        weight_elev = np.exp(-(elev**2) / (2*elev_sigma**2))
        weight = weight_azim * weight_elev
        
        mask_beam = np.random.rand(len(pts)) < weight
        pts = pts[mask_beam]
        r = r[mask_beam]
        azim = azim[mask_beam]
        elev = elev[mask_beam]
        weight = weight[mask_beam]
        
        if len(pts) == 0:
            return
        
        # 5) Açısal Çözünürlük (~15°)
        azim_bins = np.arange(-azim_fov, azim_fov, np.radians(15))
        azim_idx = np.digitize(azim, azim_bins)
        
        elev_bins = np.arange(-elev_fov, elev_fov, np.radians(15))
        elev_idx = np.digitize(elev, elev_bins)
        
        combined_idx = azim_idx * 100 + elev_idx
        _, unique_idx = np.unique(combined_idx, return_index=True)
        
        pts = pts[unique_idx]
        r = r[unique_idx]
        azim = azim[unique_idx]
        elev = elev[unique_idx]
        weight = weight[unique_idx]
        
        # 6) Range quantization (4cm)
        r_quantized = np.round(r / 0.04) * 0.04
        
        # 7) Maksimum nokta sayısı (128)
        if len(pts) > 128:
            scores = (1.0 / (r + 1.0)) * weight
            top_idx = np.argsort(-scores)[:128]
            pts = pts[top_idx]
            r = r[top_idx]
            azim = azim[top_idx]
            elev = elev[top_idx]
        
        # 8) Radar noise (mesafeye bağlı)
        range_noise_std = 0.04 + 0.001 * r
        angle_noise_std = np.radians(1.0 + 0.05 * r)
        
        r_noisy = np.abs(r + np.random.normal(0, range_noise_std))
        azim_noisy = azim + np.random.normal(0, angle_noise_std)
        elev_noisy = elev + np.random.normal(0, angle_noise_std)
        
        # Koordinatlara geri dönüştür
        pts[:,0] = r_noisy * np.cos(elev_noisy) * np.cos(azim_noisy)
        pts[:,1] = r_noisy * np.cos(elev_noisy) * np.sin(azim_noisy)
        pts[:,2] = r_noisy * np.sin(elev_noisy)
        
        # 9) RCS-based intensity
        base_intensity = np.exp(-r / 20.0)
        rcs_variation = np.random.uniform(0.5, 1.5, len(pts))
        intensity = base_intensity * rcs_variation
        
        # SNR threshold
        snr = 20 * np.log10(intensity / 0.01)
        mask_snr = snr > -10.0
        
        pts = pts[mask_snr]
        r_noisy = r_noisy[mask_snr]
        azim_noisy = azim_noisy[mask_snr]
        elev_noisy = elev_noisy[mask_snr]
        intensity = intensity[mask_snr]
        
        if len(pts) == 0:
            return
        
        # 10) Build RadarPoints message
        out = RadarPoints()
        out.header = msg.header
        out.header.frame_id = "radar_link"
        
        for i, p in enumerate(pts):
            rp = RadarPoint()
            rp.x = float(p[0])
            rp.y = float(p[1])
            rp.z = float(p[2])
            rp.intensity = float(intensity[i])
            rp.range = float(r_noisy[i])
            rp.azimuth = float(azim_noisy[i])
            rp.elevation = float(elev_noisy[i])
            out.points.append(rp)
        
        self.pub.publish(out)
        
        # Debug log (opsiyonel)
        if len(pts) > 0:
            self.get_logger().debug(
                f"[{self.platform_type.upper()}] h={height:.2f}m | "
                f"Published {len(pts)} points"
            )

def main(args=None):
    rclpy.init(args=args)
    node = RadarFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()