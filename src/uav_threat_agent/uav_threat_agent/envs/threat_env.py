import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time
import cv2

class ThreatAgentEnv(gym.Env):
    """
    Multimodal Girişli (BEV Görüntü + Durum Vektörü) Tehdit Sınıflandırma Ortamı.
    Gelişmiş Ödül Fonksiyonu (TTC + False Alarm Cezası) ile güncellendi.
    """
    
    def __init__(self):
        super(ThreatAgentEnv, self).__init__()
        
        # --- 1. ACTION SPACE (Çıktılar) ---
        # Top-K (5) nesne için 0.0 ile 1.0 arasında risk skoru
        self.K = 5
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.K,), dtype=np.float32)

        # --- 2. OBSERVATION SPACE (Girdiler) ---
        self.observation_space = spaces.Dict({
            # BEV Görüntüsü: 3 Kanal, 64x64 piksel (0-1 arası normalize)
            "bev_image": spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32),
            # Durum Vektörü: 88 elemanlı
            "state_vector": spaces.Box(low=-np.inf, high=np.inf, shape=(88,), dtype=np.float32)
        })

        # --- 3. ROS 2 BAĞLANTILARI ---
        if not rclpy.ok():
            rclpy.init()
            
        self.node = rclpy.create_node('gym_threat_interface')
        self._running = True
        self.br = CvBridge()
        
        # Veri Saklama Alanları
        self.latest_image = np.zeros((3, 64, 64), dtype=np.float32)
        self.latest_vector = np.zeros((88,), dtype=np.float32)
        self.new_img = False
        self.new_vec = False
        self.cond = threading.Condition()

        # Abonelikler
        self.sub_vec = self.node.create_subscription(
            Float32MultiArray, 
            '/threat/state_vec', 
            self.vec_callback, 
            10
        )
        
        self.sub_img = self.node.create_subscription(
            Image, 
            '/bev/image', 
            self.img_callback, 
            10
        )

        self.thread = threading.Thread(target=self._spin_node, daemon=True)
        self.thread.start()
        time.sleep(1.0)

    def _spin_node(self):
        self._running = True
        while self._running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def vec_callback(self, msg):
        try:
            data = np.array(msg.data, dtype=np.float32)
            if data.shape[0] == 88:
                with self.cond:
                    self.latest_vector = data
                    self.new_vec = True
                    self.cond.notify_all()
            else:
                self.node.get_logger().warn(f"Beklenmeyen vektör boyutu: {data.shape[0]}, beklenen: 88")
        except Exception as e:
            self.node.get_logger().error(f"Vector callback hatası: {e}")

    def img_callback(self, msg):
        try:
            cv_img = self.br.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            # 64x64 Resize işlemi
            cv_resized = cv2.resize(cv_img, (64, 64))
            
            # Normalize et (0-255 -> 0.0-1.0) ve Transpose (H,W,C -> C,H,W)
            img_transposed = np.transpose(cv_resized, (2, 0, 1)).astype(np.float32) / 255.0
            
            with self.cond:
                self.latest_image = img_transposed
                self.new_img = True
                self.cond.notify_all()
        except Exception as e:
            self.node.get_logger().error(f"Image callback hatası: {e}")

    def _wait_for_obs(self, timeout=1.0):
        end = time.time() + timeout
        with self.cond:
            while time.time() < end:
                if self.new_img and self.new_vec:
                    self.new_img = False
                    self.new_vec = False
                    return True
                remaining = end - time.time()
                if remaining > 0:
                    self.cond.wait(timeout=remaining)
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._wait_for_obs(timeout=1.0)
        observation = {
            "bev_image": self.latest_image,
            "state_vector": self.latest_vector
        }
        return observation, {}

    def step(self, action):
        # Yeni veriyi bekle
        self._wait_for_obs(timeout=0.2)
        
        obs = {
            "bev_image": self.latest_image,
            "state_vector": self.latest_vector
        }
        
        # --- ÖDÜL HESAPLAMA ---
        reward = self.calculate_reward(action, obs)
        
        terminated = False 
        truncated = False 
        
        return obs, reward, terminated, truncated, {}

    def calculate_reward(self, action, obs):
        """
        Makale tabanlı Gelişmiş Ödül Fonksiyonu:
        1. Time-to-Collision (TTC) düşükse ve Ajan yüksek skor verdiyse -> BÜYÜK ÖDÜL (+1)
        2. Tehlike yokken Ajan yüksek skor verdiyse (False Alarm) -> CEZA (-0.5)
        3. Tehlike varken düşük skor verdiyse (Miss) -> BÜYÜK CEZA (-1.0)
        """
        total_reward = 0.0
        
        vector = obs["state_vector"]
        # İlk 3 eleman UAV durumu (hız, heading), sonraki 85 eleman objeler
        objects_flat = vector[3:] 
        
        for i in range(self.K):
            start_idx = i * 17
            obj_data = objects_flat[start_idx : start_idx + 17]
            
            # Vektör İndeksleri:
            # 4: Range (Mesafe)
            # 5: Closing Speed (Yaklaşma Hızı)
            # 16: Is Valid (Geçerli obje mi?)
            
            dist = obj_data[4]
            closing_speed = obj_data[5]
            is_valid = obj_data[16]
            
            agent_score = action[i]  # Ajanın bu objeye verdiği 0.0-1.0 arası puan
            
            if is_valid > 0.5:
                # --- A. GERÇEK RİSK ANALİZİ (Ground Truth) ---
                # Time to Collision (TTC) hesabı: Mesafe / Yaklaşma Hızı
                # Eğer closing_speed <= 0 ise obje uzaklaşıyordur (TTC sonsuz)
                
                is_dangerous = False
                
                if closing_speed > 0.1: # Sadece yaklaşanlara bak
                    ttc = dist / (closing_speed + 1e-5)
                    # Eğer çarpışmaya 3 saniyeden az kaldıysa VEYA çok yakındaysa (1.5m)
                    if ttc < 3.0 or dist < 1.5:
                        is_dangerous = True

                # --- B. ÖDÜL / CEZA MANTIĞI ---
                
                if is_dangerous:
                    # DURUM 1: Tehlike VAR
                    if agent_score > 0.6: 
                        # Ajan bildi (+1.0 Ödül - Makaledeki 'Hit Reward')
                        total_reward += 1.0
                    else:
                        # Ajan tehlikeyi kaçırdı (-1.0 Ceza - Kritik Hata)
                        # Skor ne kadar düşükse ceza o kadar artar
                        total_reward -= (1.0 - agent_score) 
                        
                else:
                    # DURUM 2: Tehlike YOK (Uzaklaşıyor veya çok uzakta)
                    if agent_score > 0.4:
                        # Ajan boş yere panikledi (-0.5 Ceza - Makaledeki 'Miss Penalty')
                        # 'Paranoyak' olmayı engeller.
                        total_reward -= 0.5 * agent_score
                    else:
                        # Ajan sakin kaldı, doğru yaptı (+0.1 Ufak Ödül)
                        total_reward += 0.1

            else:
                # DURUM 3: Boş Slot (Padding)
                # Ajan buraya 0 vermeli. Eğer 0 vermezse ceza ver.
                total_reward -= agent_score * 0.5 # Gereksiz aktivasyon cezası

        # Toplam ödülü normalize edebilirsin (Opsiyonel, şimdilik ham kalsın)
        return total_reward

    def close(self):
        self._running = False
        time.sleep(0.2)
        self.node.destroy_node()
        rclpy.shutdown()