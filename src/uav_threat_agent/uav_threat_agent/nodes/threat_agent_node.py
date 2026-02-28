import gymnasium as gym
import uav_threat_agent
from stable_baselines3 import PPO
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import numpy as np
import os
import json

class ThreatPublisher(Node):
    def __init__(self):
        super().__init__('threat_agent_publisher')
        # 1. Ham Skorlar (0.0 - 1.0 arası)
        self.score_pub = self.create_publisher(Float32MultiArray, '/threat/output_scores', 10)
        # 2. Detaylı JSON Bilgisi (ID, Sınıf, Hız vb.)
        self.info_pub = self.create_publisher(String, '/threat/detailed_info', 10)

def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)

    print("--- GELİŞMİŞ AJAN (V7) YÜKLENİYOR... ---")

    # 1. Ortamı Oluştur (V2 olmasına dikkat!)
    env = gym.make('ThreatAgent-v11')
    obs, _ = env.reset()

    # 2. MODELİ YÜKLE
    # Yeni eğiteceğin model buraya düşecek (isim değişebilir, kontrol et)
    # Örn: models/PPO-1/10240.zip gibi
    model_path = "/home/ubuntu/Desktop/ros2_env/models/PPO-11-20260225-071128_resume_20260225-090502/ppo_threat_45056_steps.zip" 
    
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        # Test için models/PPO klasöründeki eski bir modeli de deneyebilirsin
        return

    try:
        model = PPO.load(model_path, env=env)
        print(f"--- MODEL BAŞARIYLA YÜKLENDİ ---")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return

    threat_pub = ThreatPublisher()
    
    try:
        while rclpy.ok():
            # A) Modelden Tahmin Al
            action, _state = model.predict(obs, deterministic=True)

            # B) Skorları Yayınla
            msg = Float32MultiArray()
            msg.data = action.tolist() 
            threat_pub.score_pub.publish(msg)

            # C) Adım At ve INFO Verisini Al
            obs, reward, terminated, truncated, info = env.step(action)

            # --- YENİ: JSON DETAYLARINI YAYINLA ---
            # Environment kodunda 'top_threats' anahtarı ile göndermiştik
            if "top_threats" in info:
                threat_data = info["top_threats"]
                
                # 1. Konsola yazdır (Okunabilir formatta)
                # print("\n--- GÜNCEL TEHDİT ANALİZİ ---")
                # print(json.dumps(threat_data, indent=2))
                
                # 2. ROS Topic'e JSON string olarak bas (Diğer nodelar okusun diye)
                json_msg = String()
                json_msg.data = json.dumps(threat_data)
                threat_pub.info_pub.publish(json_msg)

            if terminated or truncated:
                obs, _ = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        print("Kapatılıyor...")
        env.close()
        threat_pub.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()