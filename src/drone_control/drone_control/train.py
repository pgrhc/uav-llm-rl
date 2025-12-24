from stable_baselines3 import PPO
from drone_env import PX4DroneEnv # Yukarıdaki dosyayı kaydettiğin isim

# 1. Ortamı oluştur
env = PX4DroneEnv()

# 2. Modeli tanımla (PPO)
# MlpPolicy: Giriş verisi resim değilse (koordinatlar vs.) kullanılır.
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_drone_tensorboard/")

# 3. Eğitimi başlat
print("Eğitim başlıyor...")
model.learn(total_timesteps=100000)

# 4. Modeli kaydet
model.save("ppo_drone_final")
print("Eğitim bitti ve model kaydedildi.")