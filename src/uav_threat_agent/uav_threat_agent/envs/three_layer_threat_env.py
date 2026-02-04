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
        V4 FINAL: Stabilize edilmiş ödül ve JSON Info üretimi.
        """
        total_reward = 0.0
        vector = obs["state_vector"]
        objects_flat = vector[3:] 
        
        valid_obj_count = 0
        detailed_threats = [] 
        
        class_map = {0: "Unknown", 1: "Drone", 2: "Bird", 3: "FixedWing", 4: "Person"}
        
        for i in range(self.K):
            start_idx = i * 17
            obj_data = objects_flat[start_idx : start_idx + 17]
            
            obj_id = int(obj_data[0])
            class_id = int(obj_data[1])
            azimuth = obj_data[2] 
            dist = obj_data[4]
            closing_speed = obj_data[5]
            is_valid = obj_data[16]
            
            current_score = float(action[i])
            prev_score = self.prev_action[i]
            prev_risk = self.prev_target_risk[i] 
            
            if is_valid > 0.5:
                valid_obj_count += 1
                
                # --- INFO / JSON HAZIRLIĞI ---
                deg = math.degrees(azimuth)
                while deg > 180: deg -= 360
                while deg < -180: deg += 360
                
                sector = "rear"
                if -45 <= deg <= 45: sector = "front"
                elif 45 < deg <= 135: sector = "left"
                elif -135 <= deg < -45: sector = "right"

                source = "lidar"
                if class_id > 0: source = "yolo_fused"
                elif dist > 15.0: source = "radar"

                threat_info = {
                    "id": obj_id,
                    "score": round(current_score, 3),
                    "rel_dist": round(float(dist), 2),
                    "rel_vel": round(float(closing_speed), 2),
                    "source": source,
                    "sector": sector,
                    "class": class_map.get(class_id, "Unknown")
                }
                detailed_threats.append(threat_info)
                
                # --- RISK HESAPLAMA ---
                dist_factor = 1.0 / (1.0 + np.exp(1.2 * (dist - 3.5)))
                
                clamped_speed = np.clip(closing_speed, -2.0, 10.0)
                speed_contribution = 0.0
                if clamped_speed > 0.1:
                    speed_contribution = np.clip(0.15 * clamped_speed, 0.0, 0.4)
                
                target_risk = np.clip(dist_factor + speed_contribution, 0.0, 1.0)
                
                # --- REWARD ---
                alignment_reward = 1.0 - abs(current_score - target_risk)
                total_reward += alignment_reward
                
                delta_risk = abs(target_risk - prev_risk)
                delta_score = abs(current_score - prev_score)
                
                if delta_score > (delta_risk + 0.1):
                    penalty = delta_score - delta_risk
                    clipped_penalty = np.clip(penalty, 0.0, 1.0) 
                    total_reward -= 0.15 * clipped_penalty
                
                if target_risk < 0.1 and current_score > 0.4:
                    total_reward -= 0.3 * current_score 

                self.prev_target_risk[i] = target_risk
                
            else:
                total_reward -= current_score * 0.5
                self.prev_target_risk[i] = 0.0

        if valid_obj_count > 0:
            total_reward = total_reward / valid_obj_count

        # Bu 'info_data' Inference node'una gidecek
        info_data = {
            "top_threats": detailed_threats,
            "valid_count": valid_obj_count
        }
            
        return total_reward, info_data

    def close(self):
        self._running = False
        time.sleep(0.2)
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()