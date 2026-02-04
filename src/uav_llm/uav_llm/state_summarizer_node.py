
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import json
import math
import numpy as np

# --- STANDART MESAJLAR ---
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Float32MultiArray

# --- VISION MESAJLARI (YOLO) ---
from vision_msgs.msg import Detection2DArray

# --- SENİN CUSTOM MESAJLARIN (RADAR & FUSION) ---
# RadarFilter kodunda "RadarPoints" kullanmışsın, onu import ediyoruz.
try:
    from fusion_msgs.msg import RadarPoints
except ImportError:
    print("HATA: fusion_msgs bulunamadı! 'colcon build' yaptığından emin ol.")

# COCO Class ID Mapping (YOLO genellikle bu ID'leri kullanır)
# LLM'e "0" demek yerine "person" demek daha iyidir.
COCO_CLASSES = {
    "0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane",
    "5": "bus", "9": "traffic light", "10": "fire hydrant", "11": "stop sign"
}

class StateSummarizerNode(Node):
    def __init__(self):
        super().__init__('state_summarizer_node')

        # --- DURUM BELLEĞİ ---
        self.state = {
            "speed": 0.0,            # m/s
            "threat_score": 0.0,     # 0.0 - 1.0
            "lidar": {               # Sektörel boşluk durumu
                "front_space": 99.0, 
                "left_space": 99.0, 
                "right_space": 99.0
            },
            "radar_nearest": 99.0,   # En yakın radar teması (metre)
            "visual_objects": []     # YOLO'dan gelen nesneler (örn: ["person", "car"])
        }

        # --- QOS AYARLARI ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- SUBSCRIBERS ---

        # 1. Odometry (Hız için)
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)

        # 2. Lidar (Mekansal Farkındalık - Sektör Analizi)
        self.create_subscription(LaserScan, '/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan', self.scan_callback, sensor_qos)

        # 3. Radar (Senin Custom Node'undan gelen veri)
        # Topic adı senin RadarFilter kodundaki ile AYNI olmalı:
        self.create_subscription(RadarPoints, '/radar/points_filtered_radarmsg', self.radar_callback, sensor_qos)

        # 4. YOLO (Görsel Algılama)
        # Topic adı senin YoloNode kodundaki ile AYNI olmalı:
        self.create_subscription(Detection2DArray, '/yolo/detections', self.yolo_callback, 10)

        # 5. Threat Score (RL Ajanından)
        self.create_subscription(Float32MultiArray, '/threat/output_scores', self.threat_callback, 10)

        # --- PUBLISHER ---
        # LLM'e gidecek saf, işlenmiş bilgi
        self.summary_pub = self.create_publisher(String, '/llm/system_summary', 10)

        # 1 Hz'de özet yayınla
        self.timer = self.create_timer(1.0, self.publish_summary)

        self.get_logger().info("✅ State Summarizer Başlatıldı (Radar+Yolo+Lidar)")

    # ---------------- CALLBACKS ----------------

    def odom_callback(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.state["speed"] = round(math.sqrt(vx**2 + vy**2), 2)

    def scan_callback(self, msg):
        """Lidar verisini LLM için 3 sektöre böler"""
        ranges = np.array(msg.ranges)
        
        # --- DÜZELTME BAŞLANGICI ---
        # 1. Tüm sonsuzları (eksi ve artı) ve NaN (hatalı) değerleri yakala
        # Simülasyonda bazen "görülemeyen" yerler -inf dönebilir, bunları "10.0" (boşluk) sayalım.
        # VEYA çok yakın mesafe hatası ise güvenli tarafta kalıp "0.1" de diyebiliriz ama
        # senin durumunda koridorun ortasındasın, muhtemelen sensör hatası veya menzil dışı.
        # Bu yüzden 10.0 (açık alan) olarak işaretlemek mantıklı.
        
        ranges = np.where(np.isinf(ranges) | np.isnan(ranges) | (ranges == 0.0), 10.0, ranges)
        # --- DÜZELTME BİTİŞİ ---
        
        if len(ranges) == 0: return

        # Basitçe 3'e böl: Sağ, Ön, Sol
        # NOT: Lidar'ın tarama yönüne göre bu dilimler (slice) yer değiştirebilir.
        # Genelde: [Sağ Arka ... Ön ... Sol Arka] şeklindedir.
        one_third = len(ranges) // 3
        
        # Sektörlerdeki en yakın engeli bul
        self.state["lidar"]["right_space"] = round(float(np.min(ranges[:one_third])), 1)
        self.state["lidar"]["front_space"] = round(float(np.min(ranges[one_third:2*one_third])), 1)
        self.state["lidar"]["left_space"]  = round(float(np.min(ranges[2*one_third:])), 1)

    def radar_callback(self, msg):
        """
        Gelen Mesaj: fusion_msgs/RadarPoints
        İçerik: list of RadarPoint (x, y, z, range, azimuth...)
        Amaç: En yakın radar temasını bulmak.
        """
        if not msg.points:
            self.state["radar_nearest"] = 99.0
            return

        # RadarPoint objesinin içinde 'range' attribute'u var (senin kodunda gördüm)
        # En küçük range değerini buluyoruz.
        min_range = min([p.range for p in msg.points])
        
        self.state["radar_nearest"] = round(float(min_range), 2)

    def yolo_callback(self, msg):
        """
        DÜZELTİLDİ: vision_msgs yapısına uygun erişim.
        """
        detected_classes = set()
        
        for detection in msg.detections:
            for result in detection.results:
                # --- DÜZELTME BURADA ---
                # result (ObjectHypothesisWithPose) -> hypothesis (ObjectHypothesis) -> score
                if result.hypothesis.score > 0.4:
                    class_id_str = result.hypothesis.class_id 
                    
                    class_name = COCO_CLASSES.get(class_id_str, f"obj_{class_id_str}")
                    detected_classes.add(class_name)

        self.state["visual_objects"] = list(detected_classes)

    def threat_callback(self, msg):
        # RL ajanı threat skoru
        if hasattr(msg, 'data') and len(msg.data) > 0:
            self.state["threat_score"] = round(float(max(msg.data)), 2)

    # ---------------- PUBLISH ----------------

    def publish_summary(self, event=None):
        msg = String()
        msg.data = json.dumps(self.state)
        self.summary_pub.publish(msg)
        
        # Log basalım ki çalıştığını görelim
        self.get_logger().info(f"Summary: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = StateSummarizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()