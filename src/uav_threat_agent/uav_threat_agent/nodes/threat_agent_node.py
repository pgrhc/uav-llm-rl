import gymnasium as gym
import uav_threat_agent
from stable_baselines3 import PPO
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import os

class ThreatPublisher(Node):
    def __init__(self):
        super().__init__('threat_agent_publisher')
        # Tehdit skorlarını yayınlayacağımız topic
        self.publisher_ = self.create_publisher(Float32MultiArray, '/threat/output_scores', 10)

def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)

    print("--- AJAN YÜKLENİYOR... ---")

    # 1. Ortamı Oluştur
    # Inference modunda 'render_mode' gerekebilir ama şimdilik düz yapalım
    env = gym.make('ThreatAgent-v0')
    obs, _ = env.reset()

    # 2. EĞİTİLMİŞ MODELİ YÜKLE
    # BURAYA DİKKAT: Ekran görüntüsündeki modelin tam yolunu yaz.
    # Eğer dosya .zip ise uzantıyı ekle, klasör ise sonuna / koyma.
    model_path = "/home/ubuntu/Desktop/ros2_env/models/PPO/20480.zip" 
    
    # Model dosyasının varlığını kontrol et
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return

    try:
        # Modeli yükle
        model = PPO.load(model_path, env=env)
        print(f"--- MODEL BAŞARIYLA YÜKLENDİ: {model_path} ---")
        print("--- CANLI TEHDİT ANALİZİ BAŞLADI ---")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return

    # Yardımcı publisher node
    threat_pub = ThreatPublisher()
    
    try:
        while rclpy.ok():
            # A) Modelden Tahmin Al (Deterministic=True -> En iyi bildiğini okur, macera aramaz)
            action, _state = model.predict(obs, deterministic=True)

            # B) Skoru Yayınla
            # action: [0.9, 0.1, 0.0, ...] gibi 5 tane sayı
            msg = Float32MultiArray()
            msg.data = action.tolist() 
            threat_pub.publisher_.publish(msg)
            
            # Konsola da basalım ki çalıştığını gör (Opsiyonel)
            # print(f"Threat Scores: {np.round(action, 2)}")

            # C) Ortamda bir adım ilerle (ROS verilerini güncellemek için şart)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()

            # CPU'yu rahatlatmak için minik uyku
            # time.sleep(0.01) 

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