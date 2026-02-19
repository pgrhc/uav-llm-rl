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
import uuid

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
        self.class_seen_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.shuffle_prob = 0.1

        # --- 3. ROS 2 BAĞLANTILARI ---
        if not rclpy.ok():
            rclpy.init()
            
        self.node_name = f'gym_threat_{str(uuid.uuid4())[:8]}'
        self.node = rclpy.create_node(self.node_name)
        
        self.node.get_logger().info(f"Node başlatıldı: {self.node_name}")
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
        # Executor kullanımı, spin_once'dan daha güvenlidir
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self.node)
        
        try:
            while self._running and rclpy.ok():
                # Timeout süresini biraz kısalttık
                executor.spin_once(timeout_sec=0.05)
        except Exception as e:
            self.node.get_logger().warn(f"Spin hatası: {e}")
        finally:
            executor.remove_node(self.node)
            # Executor'ı kapatmak bazen sorun çıkarabilir, 
            # node destroy edildiğinde zaten temizlenir.

    # --- YARDIMCI: SIRALAMA (SLOT STABILITY) ---
    # --- YARDIMCI: SIRALAMA & KARIŞTIRMA (FIX 3: GENERALIZATION) ---
    def _sort_objects(self, vector):
        """
        Objeleri normalde mesafeye göre sıralar (Slot 0 = En Yakın).
        ANCAK: Aşırı uyumu (overfitting) engellemek için %30 ihtimalle rastgele karıştırır.
        Böylece ajan slot numarasına değil, 'dist' feature'ına bakmayı öğrenir.
        """
        try:
            uav_data = vector[:3] 
            objects_flat = vector[3:]
            num_objs = 5
            feat_len = 17
            
            objs_matrix = objects_flat.reshape(num_objs, feat_len)
            
            # --- FIX 3: SHUFFLE MODE ---
            # Rastgele bir sayı at, eğer shuffle_prob'dan küçükse karıştır.
            if np.random.rand() < self.shuffle_prob:
                sorted_indices = np.random.permutation(num_objs) # Rastgele Karıştırma
            else:
                sorted_indices = np.argsort(objs_matrix[:, 4])   # Mesafeye Göre Sıralama
            
            sorted_objs = objs_matrix[sorted_indices]
            
            return np.concatenate((uav_data, sorted_objs.flatten()))
        except Exception:
            return vector

    # --- CALLBACKS ---
    def vec_callback(self, msg):
        try:
            data = np.array(msg.data, dtype=np.float32)
            if data.shape[0] == 88:
                data = self._filter_stale_objects(data)
                sorted_data = self._sort_objects(data)
                with self.cond:
                    self.latest_vector = sorted_data
                    self.new_vec = True
                    self.cond.notify_all()
        except Exception:
            pass

    def _filter_stale_objects(self, vector):
        """
        Stale/phantom objeleri temizle.
        Kural: dist > 15m VE speed ≈ 0 → muhtemelen hayalet, is_valid=0 yap.
        """
        objects_flat = vector[3:].reshape(5, 17).copy()
        
        for i in range(5):
            dist         = objects_flat[i][4]
            closing_speed = abs(objects_flat[i][5])
            is_valid     = objects_flat[i][16]
            
            if is_valid > 0.5:
                # Çok uzak VE hareket etmiyorsa → stale data
                if dist > 15.0 and closing_speed < 0.2:
                    objects_flat[i][16] = 0.0  # is_valid = False yap
                    
        vector[3:] = objects_flat.flatten()
        return vector

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
        # self.node.get_logger().warn(
        #     "Sensor timeout! Stale data kullanılıyor. "
        #     f"(vec={self.new_vec}, lidar={self.new_lidar}, "
        #     f"radar={self.new_radar}, yolo={self.new_yolo})"
        # )
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
        V9.4 ULTIMATE (All Exploits Patched):
        - Fix A: Ghost Threat cezası (2.0) -> Hayaletleri engeller.
        - Fix B: Sparsity cezası (Power 1.5) -> Gereksiz skorlamayı engeller.
        - Fix C: Smoothness (Continuous) -> 'Gecikmeli Ajan'ı engeller.
        - Fix D: Confidence Boosting -> Drone/Bird ayrımını öğretir.
        - Fix E: Reward Scaling -> Gradient patlamasını engeller.
        - Fix F: Metrics & Class Counting -> Debugging sağlar.
        - Fix G: Critical Miss Penalty -> 'Sessiz Ajan'ı (Risk > 0.8 iken susanı) döver.
        """
        total_reward = 0.0
        vector = obs["state_vector"]
        objects_flat = vector[3:] 
        
        valid_obj_count = 0
        detailed_threats = [] 
        
        # Sınıf Risk Haritası
        # class_risk_map = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.6}
        class_names = {0: "Unknown", 1: "Drone", 2: "Bird", 3: "FixedWing", 4: "Person"}
        
        action_sum = 0.0
        
        # Temporal Smoothing için Alpha
        alpha = 0.3 
        
        # Metrikler için veri toplama listeleri
        all_target_risks = []
        all_actions = []

        for i in range(self.K):
            start_idx = i * 17
            obj_data = objects_flat[start_idx : start_idx + 17]
            
            obj_id = int(obj_data[0])
            class_id = int(obj_data[1])
            dist = obj_data[4]
            closing_speed = obj_data[5]
            is_valid = obj_data[16]

            if class_id not in [0, 1, 2, 3, 4]:
                is_valid = 0.0  # Zorla geçersiz yap, reward hesabına girmesin

            if class_id == 0:
                if closing_speed > 0.3:
                    c_factor = 0.7   # Hareket ediyor → YOLO'nun kaçırdığı insan
                elif closing_speed > 0.1:
                    c_factor = 0.3   # Yavaş hareket → Belirsiz, orta risk
                else:
                    c_factor = 0.05  # Statik → Duvar, costmap halleder

            elif class_id == 4:  # Person
                c_factor = 0.6   # YOLO gördü, kesin insan

            else:
                # Drone/Bird/FixedWing maze'de yok
                # Geliyorsa stale data veya hatalı publisher
                c_factor = 0.0

            current_score = float(action[i])
            prev_score = self.prev_action[i]
            prev_risk = self.prev_target_risk[i] 
            
            if is_valid > 0.5:
                valid_obj_count += 1
                action_sum += current_score
                
                # --- FIX F: DATA IMBALANCE TRACKING ---
                if class_id in self.class_seen_counts:
                    self.class_seen_counts[class_id] += 1
                
                # --- TARGET RISK HESABI ---
                dist_score = 1.0 / (1.0 + np.exp(1.5 * (dist - 2.5)))
                
                speed_score = 0.0
                if closing_speed > 0.1:
                    speed_score = np.clip(0.3 * closing_speed, 0.0, 0.8)
                
                raw_risk = np.clip(dist_score + speed_score, 0.0, 1.0)                
                instant_target_risk = raw_risk * c_factor  # ← Artık doğru c_factor
                
                # Temporal Smoothing (EMA)
                target_risk = alpha * instant_target_risk + (1 - alpha) * prev_risk
                
                # Metrikler için kaydet
                all_target_risks.append(target_risk)
                all_actions.append(current_score)
                
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
                
                # --- 1. ALIGNMENT REWARD ---
                diff = abs(current_score - target_risk)
                base_penalty = diff * 1.5
                
                # Polynomial Gap Penalty
                penalty = base_penalty * (1 + (diff / 0.4) ** 2)
                penalty = min(penalty, 2.5) 
                
                alignment_reward = 1.0 - penalty
                total_reward += alignment_reward
                
                # --- FIX G: CRITICAL MISS PENALTY (Sessiz Ajan Fix) ---
                # Eğer risk çok yüksek (>0.8) ama ajan uyuyorsa (<0.4), ekstra ceza!
                if target_risk > 0.5 and current_score < 0.3:
                    total_reward -= 1.5 # Çok ağır ceza, uyanması lazım.

                # --- 2. CONFIDENCE BOOSTING (Drone vs Bird) ---
                confidence_bonus = 0.0
                # if class_id == 1 and current_score > 0.8: # Drone + Yüksek Skor
                #     confidence_bonus = 0.1
                # elif class_id == 2 and current_score < 0.3: # Kuş + Düşük Skor
                #     confidence_bonus = 0.05
                # Maze için sadece Person confidence boost:
                if class_id == 4 and current_score > 0.6:   # Person + Yüksek Skor
                    confidence_bonus = 0.1
                elif class_id == 0 and closing_speed < 0.1 and current_score < 0.1:  # Duvar + Düşük Skor
                    confidence_bonus = 0.05
                
                total_reward += confidence_bonus
                
                # --- FIX C: SMOOTHNESS FIX (Gecikmeli Ajan Fix) ---
                delta_score = abs(current_score - prev_score)
                # Sadece 0.2 üstünü değil, her değişimi (0.03) ve büyükleri (0.1) cezalandır.
                # Bu sayede ajan "0.19 artırarak kaçayım" diyemez.
                smooth_penalty = 0.03 * delta_score + 0.1 * max(0, delta_score - 0.15)
                total_reward -= smooth_penalty
                
                self.prev_target_risk[i] = target_risk
                
            else:
                # --- FIX A: GHOST THREAT FIX ---
                # Geçersiz slota skor basmanın cezası.
                total_reward -= current_score * 2.0 
                self.prev_target_risk[i] = 0.0
                # Metrikler için
                all_target_risks.append(0.0)
                all_actions.append(current_score)

        # --- FIX B: SPARSITY PENALTY (GÜÇLENDİRİLMİŞ) ---
        if valid_obj_count > 0:
            avg_action = action_sum / valid_obj_count
            # Power 1.5 kullanılarak "her şeye 0.4 basma" stratejisi engellendi.
            sparsity_cost = 0.15 * (avg_action ** 1.5)
            total_reward -= sparsity_cost
            
            # --- FIX E: REWARD SCALING FIX ---
            # Valid count'a bölme KALDIRILDI. Clipping yapılıyor.
        total_reward = np.clip(total_reward, -3.0, 3.0) # Limitler biraz genişletildi
            
        # --- FIX F: METRICS GENERATION ---
        arr_actions = np.array(all_actions)
        arr_targets = np.array(all_target_risks)
        
        metrics = {
            "mean_score_var": float(np.var(arr_actions)),
            "temporal_jitter": float(np.mean(np.abs(arr_actions - self.prev_action))),
            "drone_seen_count": self.class_seen_counts[1],
            "bird_seen_count": self.class_seen_counts[2]
        }
        
        # High Risk Coverage
        high_risk_mask = arr_targets > 0.4
        if np.any(high_risk_mask):
            coverage = np.sum(arr_actions[high_risk_mask] > 0.6) / np.sum(high_risk_mask)
            metrics["high_risk_coverage"] = float(coverage)
        else:
            metrics["high_risk_coverage"] = 0.0

        # Info paketini güncelle
        info_data = {
            "top_threats": detailed_threats,
            "metrics": metrics
        }
            
        return total_reward, info_data
    def close(self):
        self._running = False
        time.sleep(0.2)
        try:
            self.node.destroy_node()
        except Exception:
            pass