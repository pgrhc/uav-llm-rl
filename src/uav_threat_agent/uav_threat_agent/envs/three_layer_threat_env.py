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
import math

class ThreatAgentEnv(gym.Env):
    """
    Advanced Threat Agent Environment (V4 Final).
    Features:
    - Distance-based Object Sorting (Slot Stability)
    - Full Sensor Synchronization
    - Stabilized Reward Function (Sigmoid Risk + Adaptive Smoothness)
    - Detailed JSON Output Generation
    """
    
    def __init__(self):
        super(ThreatAgentEnv, self).__init__()
        
        # --- 1. ACTION SPACE (DEĞİŞTİRME! Ajan sadece sayı üretir) ---
        self.K = 5
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.K,), dtype=np.float32)

        # --- 2. OBSERVATION SPACE ---
        self.observation_space = spaces.Dict({
            "bev_image": spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32),
            "state_vector": spaces.Box(low=-np.inf, high=np.inf, shape=(88,), dtype=np.float32)
        })

        # --- HAFIZA ---
        self.prev_action = np.zeros(self.K, dtype=np.float32)
        self.prev_target_risk = np.zeros(self.K, dtype=np.float32) 

        # --- 3. ROS 2 BAĞLANTILARI ---
        if not rclpy.ok():
            rclpy.init()
            
        self.node = rclpy.create_node('gym_threat_interface')
        self._running = True
        self.br = CvBridge()
        
        # Veri Saklama
        self.bev_stack = np.zeros((3, 64, 64), dtype=np.float32)
        self.latest_vector = np.zeros((88,), dtype=np.float32)
        
        # Flagler
        self.new_vec = False
        self.new_lidar = False 
        self.new_radar = False
        self.new_yolo = False
        self.cond = threading.Condition()

        # Abonelikler
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

    # --- YARDIMCI: SIRALAMA (SLOT STABILITY) ---
    def _sort_objects(self, vector):
        """Objeleri mesafeye göre sıralar. Slot 0 = En Yakın."""
        try:
            uav_data = vector[:3] 
            objects_flat = vector[3:]
            num_objs = 5
            feat_len = 17
            
            objs_matrix = objects_flat.reshape(num_objs, feat_len)
            
            # Mesafeye (Index 4) göre sırala
            sorted_indices = np.argsort(objs_matrix[:, 4])
            sorted_objs = objs_matrix[sorted_indices]
            
            return np.concatenate((uav_data, sorted_objs.flatten()))
        except Exception:
            return vector

    # --- CALLBACKS ---
    def vec_callback(self, msg):
        try:
            data = np.array(msg.data, dtype=np.float32)
            if data.shape[0] == 88:
                sorted_data = self._sort_objects(data)
                with self.cond:
                    self.latest_vector = sorted_data
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
            self.new_radar = True
            self.cond.notify_all()

    def yolo_callback(self, msg):
        img = self._process_image(msg)
        with self.cond:
            self.bev_stack[2] = img 
            self.new_yolo = True
            self.cond.notify_all()

    # --- GYM FUNCTIONS ---
    def _wait_for_obs(self, timeout=0.5):
        end = time.time() + timeout
        with self.cond:
            while time.time() < end:
                all_fresh = self.new_vec and self.new_lidar and self.new_radar and self.new_yolo
                if all_fresh:
                    self.new_vec = False
                    self.new_lidar = False
                    self.new_radar = False
                    self.new_yolo = False
                    return True
                remaining = end - time.time()
                if remaining > 0:
                    self.cond.wait(timeout=remaining)
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_action = np.zeros(self.K, dtype=np.float32)
        self.prev_target_risk = np.zeros(self.K, dtype=np.float32)
        
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
        
        # 1. Ödül Hesapla ve JSON Bilgisini (Info) Üret
        reward, info_data = self.calculate_reward_and_info(action, obs)
        
        # 2. Hafızayı Güncelle
        self.prev_action = action.copy()
        
        terminated = False 
        truncated = False 
        
        return obs, reward, terminated, truncated, info_data

    def calculate_reward_and_info(self, action, obs):
        """
        V9 CONTEXT & STABILITY (Akıllı Tehdit Analisti):
        - Class Scaling: Düşük riskli sınıfların sinyali güçlendirildi (Vanishing Gradient Fix).
        - Soft Gap Penalty: Kırılgan ceza yerine lineer artan ceza (Stability Fix).
        - Sparsity & Smoothness: L1 vergisi ve titreşim önleyici geri geldi.
        - Robust Speed: Hız katkısı netleştirildi.
        """
        total_reward = 0.0
        vector = obs["state_vector"]
        objects_flat = vector[3:] 
        
        valid_obj_count = 0
        detailed_threats = [] 
        
        # --- 1. SINIF RİSK AYARLARI (Gradient Sinyalini Koru) ---
        # 0.1 yerine 0.2-0.3 bandına çektik ki ajan "farketmez" demesin.
        # Unknown (0) = 0.7 (Belirsizlik risktir, dikkatli ol)
        # Drone (1)   = 1.0 (TAM TEHDİT)
        # Bird (2)    = 0.25 (Düşük ama ihmal edilemez)
        # FixedWing(3)= 1.0 (TAM TEHDİT)
        # Person (4)  = 0.3 (İnsan hayatı önemlidir, çarpma)
        class_risk_map = {0: 0.7, 1: 1.0, 2: 0.25, 3: 1.0, 4: 0.3}
        class_names = {0: "Unknown", 1: "Drone", 2: "Bird", 3: "FixedWing", 4: "Person"}
        
        # Sparsity (Cimrilik) için toplam aksiyon
        action_sum = 0.0
        
        for i in range(self.K):
            start_idx = i * 17
            obj_data = objects_flat[start_idx : start_idx + 17]
            
            # [0]=ID, [1]=Class, [2]=Azimuth, [4]=Range, [5]=Speed, [16]=IsValid
            obj_id = int(obj_data[0])
            class_id = int(obj_data[1])
            dist = obj_data[4]
            closing_speed = obj_data[5] # Pozitif = Yaklaşıyor
            is_valid = obj_data[16]
            
            current_score = float(action[i])
            prev_score = self.prev_action[i]
            prev_risk = self.prev_target_risk[i] 
            
            if is_valid > 0.5:
                valid_obj_count += 1
                action_sum += current_score
                
                # --- 2. HEDEF RİSK (FİZİKSEL + BAĞLAMSAL) ---
                
                # A) Mesafe (Sigmoid)
                # Merkez: 2.5m. 
                dist_score = 1.0 / (1.0 + np.exp(1.5 * (dist - 2.5)))
                
                # B) Hız (Closing Speed)
                # Sadece yaklaşıyorsa (speed > 0) risk ekle.
                # Uzaklaşan (speed < 0) cisim riski düşürmez (güvenlik payı).
                speed_score = 0.0
                if closing_speed > 0.1:
                    # Hız 5 m/s ise -> 0.3 * 5 = 1.5 (Max 0.8)
                    speed_score = np.clip(0.3 * closing_speed, 0.0, 0.8)
                
                # Ham Fiziksel Risk (Mesafe + Hız) -> Max 1.0
                raw_risk = np.clip(dist_score + speed_score, 0.0, 1.0)
                
                # C) Sınıf Çarpanı (Context)
                # Örnek: Kuş (0.25) * Fiziksel Risk (1.0) = Target (0.25)
                # Örnek: Drone (1.0) * Fiziksel Risk (1.0) = Target (1.0)
                c_factor = class_risk_map.get(class_id, 0.7)
                target_risk = raw_risk * c_factor
                
                # --- INFO ---
                threat_info = {
                    "id": obj_id,
                    "cls": class_names.get(class_id, "?"),
                    "dist": round(float(dist), 1),
                    "vel": round(float(closing_speed), 1),
                    "score": round(current_score, 2),
                    "TRGT": round(float(target_risk), 2)
                }
                detailed_threats.append(threat_info)
                
                # --- 3. ÖDÜL MEKANİZMASI (DÜZELTİLMİŞ) ---
                
                diff = abs(current_score - target_risk)
                
                # A) Temel Ceza (1.5x)
                penalty = diff * 1.5
                
                # B) Soft Gap Penalty (YUMUŞATILDI) 🛠️
                # Eğer fark > 0.4 ise, aşan kısım için ekstra ceza.
                # Örnek: Fark 0.5 ise -> (0.5 - 0.4) * 2.0 = 0.2 ekstra ceza.
                # Örnek: Fark 0.9 ise -> (0.9 - 0.4) * 2.0 = 1.0 ekstra ceza.
                # Bu fonksiyon süreklidir (kırılma yaratmaz).
                if diff > 0.4:
                    penalty += (diff - 0.4) * 2.0
                
                # Cezayı Tavanla (Reward Explosion Önlemi)
                penalty = min(penalty, 1.5)
                
                alignment_reward = 1.0 - penalty
                total_reward += alignment_reward
                
                # C) Smoothness (Hafifletilmiş)
                # Titremeyi önlemek için geri geldi.
                delta_score = abs(current_score - prev_score)
                if delta_score > 0.2: # Sadece büyük zıplamalarda
                     total_reward -= 0.05 * delta_score
                
                self.prev_target_risk[i] = target_risk
                
            else:
                # Valid değilse skor basma cezası
                total_reward -= current_score * 0.5 
                self.prev_target_risk[i] = 0.0

        # --- 4. SPARSITY (L1 VERGİSİ) ---
        # "Her şeye 1 basayım"cıları engellemek için geri geldi.
        if valid_obj_count > 0:
            avg_action = action_sum / valid_obj_count
            total_reward -= 0.05 * avg_action # Küçük bir vergi
            
            total_reward = total_reward / valid_obj_count
            
        return total_reward, {"top_threats": detailed_threats}
    
    def close(self):
        self._running = False
        time.sleep(0.2)
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()