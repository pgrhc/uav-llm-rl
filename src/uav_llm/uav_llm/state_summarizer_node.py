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

# --- CUSTOM MESAJLAR ---
try:
    from fusion_msgs.msg import RadarPoints
except ImportError:
    pass

# COCO Class ID Mapping
COCO_CLASSES = {
    "0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane",
    "5": "bus", "9": "traffic light", "10": "fire hydrant", "11": "stop sign"
}

class StateSummarizerNode(Node):
    def __init__(self):
        super().__init__('state_summarizer_node')

        # --- DURUM BELLEĞİ ---
        self.state = {
            "speed": 0.0,
            "lidar": {
                "front_space": 99.0, 
                "left_space": 99.0, 
                "right_space": 99.0
            },
            "radar_nearest": 99.0,
            "visual_objects": [],
            "primary_threat": {
                "class": "None",   
                "dist": 0.0,       
                "risk_score": 0.0, 
                "id": -1
            }
        }

        # --- QOS AYARLARI ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- SUBSCRIBERS ---

        # 1. Odometry
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)

        # 2. Lidar
        self.create_subscription(LaserScan, '/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan', self.scan_callback, sensor_qos)

        # 3. Radar
        self.create_subscription(RadarPoints, '/radar/points_filtered_radarmsg', self.radar_callback, sensor_qos)

        # 4. YOLO
        self.create_subscription(Detection2DArray, '/yolo/detections', self.yolo_callback, 10)

        # --- 5. TEHDİT AJANI (KRİTİK DÜZELTME BURADA) ---
        # HATA BURADAYDI: Eskiden '/threat/output_scores' dinliyordun, bu Float taşıyordu.
        # DOĞRUSU: '/threat/detailed_info' (JSON String taşıyan topic) olmalı.
        
        self.create_subscription(String, '/threat/detailed_info', self.threat_callback, 10)

        # --- PUBLISHER ---
        self.summary_pub = self.create_publisher(String, '/llm/system_summary', 10)
        self.timer = self.create_timer(1.0, self.publish_summary)

        self.get_logger().info("✅ State Summarizer Başlatıldı (Topic: /threat/detailed_info)")

    # ---------------- CALLBACKS ----------------

    def odom_callback(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.state["speed"] = round(math.sqrt(vx**2 + vy**2), 2)

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges) | np.isnan(ranges) | (ranges == 0.0), 10.0, ranges)
        
        if len(ranges) == 0: return

        one_third = len(ranges) // 3
        self.state["lidar"]["right_space"] = round(float(np.min(ranges[:one_third])), 1)
        self.state["lidar"]["front_space"] = round(float(np.min(ranges[one_third:2*one_third])), 1)
        self.state["lidar"]["left_space"]  = round(float(np.min(ranges[2*one_third:])), 1)

    def radar_callback(self, msg):
        if not msg.points:
            self.state["radar_nearest"] = 99.0
            return
        min_range = min([p.range for p in msg.points])
        self.state["radar_nearest"] = round(float(min_range), 2)

    def yolo_callback(self, msg):
        detected_classes = set()
        for detection in msg.detections:
            for result in detection.results:
                if result.hypothesis.score > 0.4:
                    class_id_str = result.hypothesis.class_id 
                    class_name = COCO_CLASSES.get(class_id_str, f"obj_{class_id_str}")
                    detected_classes.add(class_name)
        self.state["visual_objects"] = list(detected_classes)

    def threat_callback(self, msg):
        """
        JSON verisi artık doğru topic'ten (/threat/detailed_info) geliyor.
        """
        try:
            threat_list = json.loads(msg.data)
            
            if not isinstance(threat_list, list) or len(threat_list) == 0:
                self.reset_threat()
                return

            # Ajanın 'score' (tahmin) değerine göre en riskliyi seçiyoruz.
            most_dangerous = max(threat_list, key=lambda x: x.get("score", 0.0))

            self.state["primary_threat"] = {
                "class": most_dangerous.get("cls", "Unknown"),
                "dist": round(most_dangerous.get("dist", 0.0), 1),
                "risk_score": round(most_dangerous.get("score", 0.0), 2),
                "id": most_dangerous.get("id", -1)
            }
            
        except json.JSONDecodeError:
            self.get_logger().warn("JSON parse hatası!")
        except Exception as e:
            self.get_logger().warn(f"Threat callback hatası: {e}")

    def reset_threat(self):
        self.state["primary_threat"] = {"class": "None", "dist": 0.0, "risk_score": 0.0, "id": -1}

    # ---------------- PUBLISH ----------------

    def publish_summary(self, event=None):
        msg = String()
        msg.data = json.dumps(self.state)
        self.summary_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = StateSummarizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()