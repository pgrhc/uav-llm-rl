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
    Multimodal Girişli Tehdit Sınıflandırma Ortamı.
    GÜNCELLEME: Mesafe Tabanlı Soft Risk + Kararlılık (Smoothness) Ödülü.
    """
    
    def __init__(self):
        super(ThreatAgentEnv, self).__init__()
        
        # --- 1. ACTION SPACE ---
        self.K = 5
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.K,), dtype=np.float32)

        # --- 2. OBSERVATION SPACE ---
        self.observation_space = spaces.Dict({
            # (3, 64, 64) -> 3 Kanal: Lidar, Radar, YOLO
            "bev_image": spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32),
            "state_vector": spaces.Box(low=-np.inf, high=np.inf, shape=(88,), dtype=np.float32)
        })

        # --- YENİ: Önceki aksiyonu saklamak için hafıza (Smoothness için) ---
        self.prev_action = np.zeros(self.K, dtype=np.float32)

        # --- 3. ROS 2 BAĞLANTILARI ---
        if not rclpy.ok():
            rclpy.init()
            
        self.node = rclpy.create_node('gym_threat_interface')
        self._running = True
        self.br = CvBridge()
        
        # Veri Saklama Alanları
        self.bev_stack = np.zeros((3, 64, 64), dtype=np.float32)
        self.latest_vector = np.zeros((88,), dtype=np.float32)
        
        self.new_vec = False
        self.new_lidar = False 
        self.cond = threading.Condition()

        # --- ABONELİKLER ---
        self.sub_vec = self.node.create_subscription(
            Float32MultiArray, '/threat/state_vec', self.vec_callback, 10)
        
        self.sub_lidar = self.node.create_subscription(
            Image, '/bev/lidar_layer', self.lidar_callback, 10)
        self.sub_radar = self.node.create_subscription(
            Image, '/bev/radar_layer', self.radar_callback, 10)
        self.sub_yolo = self.node.create_subscription(
            Image, '/bev/yolo_layer', self.yolo_callback, 10)

        self.thread = threading.Thread(target=self._spin_node, daemon=True)
        self.thread.start()
        time.sleep(1.0)

    def _spin_node(self):
        self._running = True
        while self._running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    # --- CALLBACKS (Değişmedi) ---
    def vec_callback(self, msg):
        try:
            data = np.array(msg.data, dtype=np.float32)
            if data.shape[0] == 88:
                with self.cond:
                    self.latest_vector = data
                    self.new_vec = True
                    self.cond.notify_all()
        except Exception:
            pass

    def _process_image(self, msg):
        try:
            cv_img = self.br.imgmsg_to_cv2(msg, desired_encoding='mono8')
            cv_resized = cv2.resize(cv_img, (64, 64))
            return cv_resized.astype(np.float32) / 255.0
        except Exception:
            return np.zeros((64, 64), dtype=np.float32)

    def lidar_callback(self, msg):
        img = self._process_image(msg)
        with self.cond:
            self.bev_stack[0] = img 
            self.new_lidar = True
            self.cond.notify_all()

    def radar_callback(self, msg):
        img = self._process_image(msg)
        with self.cond:
            self.bev_stack[1] = img 

    def yolo_callback(self, msg):
        img = self._process_image(msg)
        with self.cond:
            self.bev_stack[2] = img 

    # --- GYM FUNCTIONS ---

    def _wait_for_obs(self, timeout=1.0):
        end = time.time() + timeout
        with self.cond:
            while time.time() < end:
                if self.new_vec and self.new_lidar:
                    self.new_vec = False
                    self.new_lidar = False
                    return True
                remaining = end - time.time()
                if remaining > 0:
                    self.cond.wait(timeout=remaining)
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # YENİ: Bölüm başında hafızayı sıfırla
        self.prev_action = np.zeros(self.K, dtype=np.float32)
        
        self._wait_for_obs(timeout=1.0)
        observation = {
            "bev_image": self.bev_stack.copy(),
            "state_vector": self.latest_vector.copy()
        }
        return observation, {}

    def step(self, action):
        self._wait_for_obs(timeout=0.2)
        
        obs = {
            "bev_image": self.bev_stack.copy(),
            "state_vector": self.latest_vector.copy()
        }
        
        # Ödül Hesapla
        reward = self.calculate_reward(action, obs)
        
        # YENİ: Bir sonraki adım için şu anki aksiyonu kaydet
        self.prev_action = action.copy()
        
        terminated = False 
        truncated = False 
        
        return obs, reward, terminated, truncated, {}

    def calculate_reward(self, action, obs):
        """
        GÜNCELLENMİŞ ÖDÜL FONKSİYONU (Path Following Aşaması İçin)
        
        1. Soft Risk: Mesafe tabanlı [0, 1] arası hedef risk.
        2. Alignment: |Ajan Skoru - Hedef Risk| farkı azaldıkça ödül.
        3. Smoothness: Skorlar aniden zıplarsa ceza.
        """
        total_reward = 0.0
        
        vector = obs["state_vector"]
        objects_flat = vector[3:] 
        
        # Parametreler
        D_CRIT = 1.5  # Bu mesafenin altı %100 risk
        D_SAFE = 4.0  # Bu mesafenin üstü %0 risk
        
        valid_obj_count = 0
        
        for i in range(self.K):
            start_idx = i * 17
            obj_data = objects_flat[start_idx : start_idx + 17]
            
            # Vektör: [4]=Range, [16]=IsValid
            dist = obj_data[4]
            is_valid = obj_data[16]
            
            current_score = action[i]
            prev_score = self.prev_action[i] # Önceki adımdaki skor
            
            if is_valid > 0.5:
                valid_obj_count += 1
                
                # --- A. HEDEF RİSKİ HESAPLA (Ground Truth) ---
                # TTC yerine daha stabil olan "Mesafe" kullanıyoruz.
                # Linear Interpolation: 1.5m -> 1.0 Risk, 4.0m -> 0.0 Risk
                
                if dist <= D_CRIT:
                    target_risk = 1.0
                elif dist >= D_SAFE:
                    target_risk = 0.0
                else:
                    # Aradaki değerler için orantı kur
                    # Örn: 2.75m ise risk 0.5 olur.
                    ratio = (D_SAFE - dist) / (D_SAFE - D_CRIT)
                    target_risk = np.clip(ratio, 0.0, 1.0)
                
                # --- B. HİZALAMA ÖDÜLÜ (Alignment Reward) ---
                # Ajanın tahmini hedef riske ne kadar yakın?
                # Tam isabetse +1.0, çok uzaksa 0.0'a yaklaşır.
                alignment_reward = 1.0 - abs(current_score - target_risk)
                total_reward += alignment_reward
                
                # --- C. KARARLILIK CEZASI (Smoothness Penalty) ---
                # Skor bir anda 0.1'den 0.9'a fırlamasın (Gürültüden kaçınma)
                jump = abs(current_score - prev_score)
                total_reward -= 0.1 * jump  # Ufak bir ceza

            else:
                # --- D. PADDING CEZASI ---
                # Boş slotlara skor verme
                total_reward -= current_score * 0.5 

        # --- E. NORMALİZASYON ---
        # Toplam ödülü valid obje sayısına böl ki 1 obje ile 5 obje arasında uçurum olmasın.
        if valid_obj_count > 0:
            total_reward = total_reward / valid_obj_count
            
        return total_reward

    def close(self):
        self._running = False
        time.sleep(0.2)
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()