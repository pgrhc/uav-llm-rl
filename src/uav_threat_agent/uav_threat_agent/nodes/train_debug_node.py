import gymnasium as gym
import uav_threat_agent  # __init__.py içindeki register kodunu tetikler
from stable_baselines3 import PPO
import os
import rclpy
import cv2

def main(args=None):
    # ROS 2 başlat (Env içinde kontrol ediyoruz ama garanti olsun)
    if not rclpy.ok():
        rclpy.init(args=args)

    print("--- EĞİTİM BAŞLATILIYOR ---")

    # 1. Ortamı Oluştur
    # id='ThreatAgent-v3' -> __init__.py dosyasında register ettiğimiz isim
    env = gym.make('ThreatAgent-v3')
    
    # 2. Kayıt Klasörlerini Ayarla
    models_dir = "models/PPO-2"
    log_dir = "logs"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 3. Modeli Tanımla
    # MultiInputPolicy: Hem Resim (CNN) hem Vektör (MLP) girdiği için şart.
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        # Policy Mimarisi: 
        # CNN otomatik feature çıkarır.
        # MLP için [256, 128] katmanları ekliyoruz.
        policy_kwargs=dict(net_arch=[256, 128]) 
    )

    # 4. Eğitimi Başlat
    # TIMESTEPS: Her kayıt öncesi kaç adım atılacağı
    TIMESTEPS = 2048 
    for i in range(1, 10):
        # Modeli eğit
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        
        # Modeli kaydet
        model.save(f"{models_dir}/{TIMESTEPS*i}")
        print(f"Model kaydedildi: {TIMESTEPS*i}. adım")

    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()