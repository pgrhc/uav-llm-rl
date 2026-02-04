import gymnasium as gym
import uav_threat_agent
from stable_baselines3 import PPO
import os
import rclpy

def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)

    print("--- PRETRAINED MODEL İLE EĞİTİM BAŞLIYOR ---")

    # 1. YENİ Ortamı Oluştur (V4 kodlu V2 Environment)
    env = gym.make('ThreatAgent-v4')
    
    # 2. Kayıt Klasörleri
    # Yeni bir klasör açalım ki eskilerle karışmasın (İsteğe bağlı)
    models_dir = "models/PPO-FineTuned-1"
    log_dir = "logs"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 3. ESKİ MODELİN YOLUNU BELİRT
    # En son kaydettiğin en iyi modelin tam yolunu buraya yaz.
    pretrained_path = "/home/ubuntu/Desktop/ros2_env/models/PPO-2/18432.zip"

    # Dosya var mı kontrol et
    if not os.path.exists(pretrained_path):
        print(f"HATA: Model bulunamadı: {pretrained_path}")
        return

    print(f"Model Yükleniyor: {pretrained_path}...")
    custom_params = {
        'learning_rate': 0.00005,  # Çok yavaş öğren (Unutmayı engeller)
        'n_epochs': 5,             # Veriyi az tekrar et (Ezberlemeyi/KL artışını engeller)
        'clip_range': 0.15,         # Değişime sıkı sınır koy (Stabilite sağlar)
        'batch_size': 128,         # Daha büyük paketlerle çalış (Gürültüyü azaltır)
        'ent_coef': 0.001           # Çok az meraklı ol (Takılmayı engeller)
    }

    # 4. MODELİ YÜKLE (LOAD)
    # env=env parametresi ŞARTTIR, yoksa yeni ortamla etkileşime giremez.
    # custom_objects: Learning Rate'i yüklerken düşürüyoruz (Stabilite için)
    model = PPO.load(
        pretrained_path, 
        env=env,
        print_system_info=True,
        # Learning Rate'i burada güncelliyoruz (Daha yavaş ve dikkatli öğrenmesi için)
        custom_objects=custom_params
    )


    # ÖNEMLİ NOT:
    # Model yüklendiğinde 'n_steps', 'batch_size' gibi yapısal parametreler 
    # eski modelden gelir (512 ve 64). Bunları değiştiremezsin çünkü
    # kayıtlı dosyanın içindeki matris boyutları buna göredir.
    # Ama Learning Rate'i değiştirmek serbesttir ve metriklerini düzeltecek olan odur.

    # Eğer manuel olarak LR değişmezse, kod içinde zorlayalım:
    model.learning_rate = 0.00005
    model.n_epochs = 5
    model.clip_range = lambda x: 0.15 # Sabit değer için lambda gerekebilir
    model.batch_size = 128
    model.ent_coef = 0.001
    print(f"Yeni Learning Rate ayarlandı: {model.learning_rate}")

    # 5. EĞİTİMİ DEVAM ETTİR
    TIMESTEPS = 2048 
    # range'i artırabilirsin, artık temel bilgi var, sadece ince ayar yapıyoruz.
    for i in range(1, 20): 
        model.learn(
            total_timesteps=TIMESTEPS, 
            reset_num_timesteps=False, # FALSE YAP! (Loglar kaldığı yerden devam etsin)
            tb_log_name="PPO_FineTune"
        )
        
        save_path = f"{models_dir}/{TIMESTEPS*i}_finetuned"
        model.save(save_path)
        print(f"Fine-Tuned Model kaydedildi: {save_path}")

    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()