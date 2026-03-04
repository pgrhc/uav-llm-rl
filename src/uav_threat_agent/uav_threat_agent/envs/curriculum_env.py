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
import json
import subprocess

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry

class ThreatAgentEnv(gym.Env):
    """
    CURRICULUM LEARNING VERSION
    3 Stages:
    - Stage 1 (0-50k): Learn to score Person high
    - Stage 2 (50k-100k): Learn distance-based scoring
    - Stage 3 (100k+): Full complexity
    """

    # Dynamic actor collision — dual-layer (YOLO + analytical)
    ACTOR_COLLISION_RADIUS = 1.5
    ACTOR_COLLISION_Z_MAX = 3.5   # actor feet=0.9, head≈2.7m + margin
    ACTOR_JSON_PATH = "/home/ubuntu/Desktop/maze_actors.json"
    COLLISION_LOG_INTERVAL = 50   # her N step'te durum logu

    # Soft reset — verify with:  gz model --list
    DRONE_MODEL_NAME = "x500_mono_cam"
    SPAWN_POSITION = (0.0, 0.0, 1.5)
    GZ_WORLD_NAME = "default"
    RESET_STABILIZE_SEC = 2.0
    MAX_EPISODE_STEPS = 500

    def __init__(self, curriculum_stage=1):
        super(ThreatAgentEnv, self).__init__()
        
        self.curriculum_stage = curriculum_stage  # 1, 2, or 3
        
        # --- 1. ACTION SPACE ---
        self.K = 5
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.K,), dtype=np.float32)

        # --- 2. OBSERVATION SPACE ---
        self.observation_space = spaces.Dict({
            "bev_image": spaces.Box(low=0.0, high=1.0, shape=(3, 128, 128), dtype=np.float32),
            "state_vector": spaces.Box(low=-np.inf, high=np.inf, shape=(88,), dtype=np.float32)
        })

        # --- HAFIZA ---
        self.prev_action = np.zeros(self.K, dtype=np.float32)
        self.prev_target_risk = np.zeros(self.K, dtype=np.float32) 
        self.class_seen_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
        # Shuffle probability based on stage
        if curriculum_stage == 1:
            self.shuffle_prob = 1.0  # Always random (no slot bias)
        elif curriculum_stage == 2:
            self.shuffle_prob = 0.7
        else:
            self.shuffle_prob = 0.7

        # --- 3. ROS 2 BAĞLANTILARI ---
        if not rclpy.ok():
            rclpy.init()
            
        self.node_name = f'gym_threat_{str(uuid.uuid4())[:8]}'
        self.node = rclpy.create_node(self.node_name)
        
        self.node.get_logger().info(f"Node başlatıldı: {self.node_name} [CURRICULUM STAGE {curriculum_stage}]")
        self._running = True
        self.br = CvBridge()
        
        # Veri Saklama
        self.bev_stack = np.zeros((3, 128, 128), dtype=np.float32)
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

        # Drone position (for collision detection)
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.step_count = 0
        self.actor_trajectories = []
        self.actor_ref_time = time.time()

        qos_odom = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.sub_odom = self.node.create_subscription(
            Odometry, '/odometry/filtered', self._cb_odom, qos_odom)

        self.thread = threading.Thread(target=self._spin_node, daemon=True)
        self.thread.start()
        time.sleep(1.0)

        self._load_actor_data()

    def _spin_node(self):
        self._running = True
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self.node)
        
        try:
            while self._running and rclpy.ok():
                executor.spin_once(timeout_sec=0.05)
        except Exception as e:
            self.node.get_logger().warn(f"Spin hatası: {e}")
        finally:
            executor.remove_node(self.node)

    def _sort_objects(self, vector):
        try:
            uav_data = vector[:3] 
            objects_flat = vector[3:]
            num_objs = 5
            feat_len = 17
            
            objs_matrix = objects_flat.reshape(num_objs, feat_len)
            
            if np.random.rand() < self.shuffle_prob:
                sorted_indices = np.random.permutation(num_objs)
            else:
                sorted_indices = np.argsort(objs_matrix[:, 4])
            
            sorted_objs = objs_matrix[sorted_indices]
            
            return np.concatenate((uav_data, sorted_objs.flatten()))
        except Exception:
            return vector

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
        objects_flat = vector[3:].reshape(5, 17).copy()
        
        for i in range(5):
            dist = objects_flat[i][4]
            closing_speed = abs(objects_flat[i][5])
            is_valid = objects_flat[i][16]
            
            if is_valid > 0.5:
                if dist > 15.0 and closing_speed < 0.2:
                    objects_flat[i][16] = 0.0
                    
        vector[3:] = objects_flat.flatten()
        return vector

    def _process_image(self, msg):
        try:
            cv_img = self.br.imgmsg_to_cv2(msg, desired_encoding='mono8')
            cv_resized = cv2.resize(cv_img, (128, 128))
            return cv_resized.astype(np.float32) / 255.0
        except Exception:
            return np.zeros((128, 128), dtype=np.float32)

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
        self.step_count = 0

        self._soft_reset_drone()
        self._load_actor_data()
        self._wait_for_obs(timeout=2.0)

        observation = {
            "bev_image": self.bev_stack.copy(),
            "state_vector": self.latest_vector.copy()
        }
        return observation, {}

    def step(self, action):
        self.step_count += 1
        self._wait_for_obs(timeout=0.2)
        
        obs = {
            "bev_image": self.bev_stack.copy(),
            "state_vector": self.latest_vector.copy()
        }
        
        # Route to appropriate reward function based on curriculum stage
        if self.curriculum_stage == 1:
            reward, info_data = self.calculate_reward_STAGE1(action, obs)
        elif self.curriculum_stage == 2:
            reward, info_data = self.calculate_reward_STAGE2(action, obs)
        else:
            reward, info_data = self.calculate_reward_STAGE3(action, obs)
        
        self.prev_action = action.copy()
        
        terminated = False
        truncated = False

        if self._check_actor_collision():
            terminated = True
            info_data["actor_collision"] = True

        if self.step_count >= self.MAX_EPISODE_STEPS:
            truncated = True
            info_data["timeout"] = True
        
        return obs, reward, terminated, truncated, info_data

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Learn to Score (0-50k)
    # ══════════════════════════════════════════════════════════════════════
    def calculate_reward_STAGE1(self, action, obs):
        """
        STAGE 1: Sadece Person'a yüksek, Unknown'a düşük skor ver
        En basit reward - Agent skor vermeyi öğrenir
        """
        total_reward = 0.0
        vector = obs["state_vector"]
        objects_flat = vector[3:]
        
        detailed_threats = []
        class_names = {0: "Unknown", 1: "Drone", 2: "Bird", 3: "FixedWing", 4: "Person"}
        
        for i in range(self.K):
            start_idx = i * 17
            obj_data = objects_flat[start_idx : start_idx + 17]
            
            obj_id = int(obj_data[0])
            class_id = int(obj_data[1])
            dist = obj_data[4]
            closing_speed = obj_data[5]
            is_valid = obj_data[16]
            current_score = float(action[i])
            
            if is_valid > 0.5:
                if class_id == 4:  # Person
                    # Person'a yüksek skor = ÖDÜL
                    reward = current_score  # 0 → 0, 1 → +1
                else:  # Unknown/Other
                    # Unknown'a düşük skor = ÖDÜL
                    reward = (1.0 - current_score)  # 0 → +1, 1 → 0
                
                total_reward += reward
                
                # Info
                threat_info = {
                    "id": obj_id,
                    "cls": class_names.get(class_id, "?"),
                    "dist": round(float(dist), 1),
                    "vel": round(float(closing_speed), 1),
                    "score": round(current_score, 2),
                    "TRGT": "STAGE1"
                }
                detailed_threats.append(threat_info)
            else:
                # Boş slota skor verme cezası
                total_reward -= current_score * 2.0
        
        total_reward = np.clip(total_reward, -10.0, 10.0)
        
        info_data = {"top_threats": detailed_threats, "metrics": {}}
        return total_reward, info_data

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Learn Distance (50k-100k)
    # ══════════════════════════════════════════════════════════════════════
    def calculate_reward_STAGE2(self, action, obs):
        """
        STAGE 2: Mesafeye göre skorlama öğren
        Yakın Person → yüksek skor, uzak Person → düşük skor
        """
        total_reward = 0.0
        vector = obs["state_vector"]
        objects_flat = vector[3:]
        
        detailed_threats = []
        class_names = {0: "Unknown", 1: "Drone", 2: "Bird", 3: "FixedWing", 4: "Person"}
        
        for i in range(self.K):
            start_idx = i * 17
            obj_data = objects_flat[start_idx : start_idx + 17]
            
            obj_id = int(obj_data[0])
            class_id = int(obj_data[1])
            dist = obj_data[4]
            closing_speed = obj_data[5]
            is_valid = obj_data[16]
            current_score = float(action[i])
            
            if is_valid > 0.5:
                if class_id == 4:  # Person
                    # Simple target risk based on distance
                    target_risk = 1.0 / (1.0 + np.exp(2.0 * (dist - 3.5)))
                    target_risk *= 0.9  # c_factor
                    
                    # Simple alignment reward
                    diff = abs(current_score - target_risk)
                    reward = 1.0 - diff  # Perfect: +1, Wrong: -1
                    
                else:  # Unknown
                    # Unknown'a düşük skor ver
                    reward = 1.0 - current_score
                
                total_reward += reward
                
                # Info
                threat_info = {
                    "id": obj_id,
                    "cls": class_names.get(class_id, "?"),
                    "dist": round(float(dist), 1),
                    "vel": round(float(closing_speed), 1),
                    "score": round(current_score, 2),
                    "TRGT": round(float(target_risk if class_id == 4 else 0.0), 2)
                }
                detailed_threats.append(threat_info)
            else:
                # Boş slota skor verme cezası
                total_reward -= current_score * 2.0
        
        total_reward = np.clip(total_reward, -10.0, 10.0)
        
        info_data = {"top_threats": detailed_threats, "metrics": {}}
        return total_reward, info_data

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: Full Complexity (100k+)
    # ══════════════════════════════════════════════════════════════════════
    def calculate_reward_STAGE3(self, action, obs):
        """
        STAGE 3: Full complexity reward function
        (Senin mevcut karmaşık reward function'ın buraya gelecek)
        """
        
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
                # ═══════════════════════════════════════════════════════════════
                # YENİ: PERSON İÇİN ÖZEL DISTANCE SCORING
                # ═══════════════════════════════════════════════════════════════
                if class_id == 4:  # Person
                    # Aggressive sigmoid: 3.5m threshold, 2.0 slope
                    dist_score = 1.0 / (1.0 + np.exp(2.0 * (dist - 3.5)))
                else:
                    # Default sigmoid (Unknown, diğerleri)
                    dist_score = 1.0 / (1.0 + np.exp(1.5 * (dist - 2.5)))
                
                # Speed score
                speed_score = 0.0
                if closing_speed > 0.1:
                    speed_score = np.clip(0.3 * closing_speed, 0.0, 0.8)
                
                raw_risk = np.clip(dist_score + speed_score, 0.0, 1.0)
                
                # ═══════════════════════════════════════════════════════════════
                # YENİ: C_FACTOR DEĞERLERİ
                # ═══════════════════════════════════════════════════════════════
                if class_id == 0:  # Unknown
                    if closing_speed > 0.3:
                        c_factor = 0.8   # Hareket ediyor
                    elif closing_speed > 0.1:
                        c_factor = 0.4   # Yavaş hareket
                    else:
                        c_factor = 0.05  # Statik (duvar)
                
                elif class_id == 4:  # Person
                    c_factor = 0.9   # ← 0.6'dan 0.9'a ARTIRILDI!
                
                else:
                    c_factor = 0.0  # Drone/Bird/FixedWing
                
                instant_target_risk = raw_risk * c_factor
                
                # Temporal Smoothing (EMA)
                target_risk = alpha * instant_target_risk + (1 - alpha) * prev_risk
                
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
                # ═══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTION FIX - three_layer_threat_env.py içine yapıştır
# ═══════════════════════════════════════════════════════════════════════════════
# SORUN: Agent "her şeye 0 ver" stratejisi öğrendi (lazy agent)
# ÇÖZÜM: 1) Critical Miss threshold düşür (0.5 → 0.25)
#        2) Person detection için pozitif ödül ekle
#        3) Confidence bonus artır
# ═══════════════════════════════════════════════════════════════════════════════

# calculate_reward_and_info() içinde, alignment_reward hesaplandıktan sonra
# aşağıdaki blokları DEĞİŞTİR:

                # --- 1. ALIGNMENT REWARD ---
                diff = abs(current_score - target_risk)
                base_penalty = diff * 1.5
                penalty = base_penalty * (1 + (diff / 0.4) ** 2)
                penalty = min(penalty, 2.5) 
                alignment_reward = 1.0 - penalty
                total_reward += alignment_reward
                
                # ═══════════════════════════════════════════════════════════════
                # YENİ: CRITICAL MISS PENALTY (Düşük Threshold)
                # ═══════════════════════════════════════════════════════════════
                if target_risk > 0.25 and current_score < 0.2:  # ← 0.5 → 0.25
                    # Gradient: risk yükseldikçe ceza artar
                    miss_penalty = 2.5 * (target_risk / 0.6)  # Max risk 0.6 için normalize
                    total_reward -= miss_penalty

                # ═══════════════════════════════════════════════════════════════
                # YENİ: POSITIVE REINFORCEMENT (Person Detection Reward)
                # ═══════════════════════════════════════════════════════════════
                if class_id == 4:  # Person
                    if current_score > 0.4:
                        # Person'a yüksek skor vermek ÖDÜL kazandırır!
                        detection_reward = 0.4 * current_score  # Max +0.4
                        total_reward += detection_reward
                
                # ═══════════════════════════════════════════════════════════════
                # YENİ: CONFIDENCE BOOSTING (Artırılmış)
                # ═══════════════════════════════════════════════════════════════
                confidence_bonus = 0.0
                if class_id == 4 and current_score > 0.6:   # Person + Yüksek Skor
                    confidence_bonus = 0.2  # ← 0.1'den 0.2'ye çıkarıldı
                elif class_id == 0 and closing_speed < 0.1 and current_score < 0.1:  # Duvar
                    confidence_bonus = 0.05
                
                total_reward += confidence_bonus
                
                # --- FIX C: SMOOTHNESS (değişiklik yok) ---
                delta_score = abs(current_score - prev_score)
                smooth_penalty = 0.03 * delta_score + 0.1 * max(0, delta_score - 0.15)
                total_reward -= smooth_penalty
                
                self.prev_target_risk[i] = target_risk

# ═══════════════════════════════════════════════════════════════════════════════
# ÖZET:
# ═══════════════════════════════════════════════════════════════════════════════
# 1. Critical Miss: target_risk > 0.25 (eskiden 0.5) → Person için tetiklenir
# 2. Person Detection Reward: score > 0.4 olunca +0.4 ödül (YENİ!)
# 3. Confidence Bonus: 0.1 → 0.2 (2 kat artırıldı)
#
# Bu değişikliklerle agent "her şeye 0" yerine Person'a yüksek skor vermeye
# teşvik edilecek, çünkü:
#   - 0 verdiğinde: Critical Miss cezası alacak (-2.5)
#   - Yüksek skor verdiğinde: Detection reward (+0.4) + Confidence bonus (+0.2)
# ═══════════════════════════════════════════════════════════════════════════════
                
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


    # ------------------------------------------------------------------ #
    # Dynamic actor collision & soft reset
    # ------------------------------------------------------------------ #
    def _cb_odom(self, msg):
        self.drone_x = msg.pose.pose.position.x
        self.drone_y = msg.pose.pose.position.y
        self.drone_z = msg.pose.pose.position.z

    def _load_actor_data(self):
        try:
            with open(self.ACTOR_JSON_PATH, "r") as f:
                data = json.load(f)
            self.actor_trajectories = data.get("actors", [])
            self.actor_ref_time = data.get("spawn_time", time.time())
            self.node.get_logger().info(
                f"Actor trajectories loaded: {len(self.actor_trajectories)}"
            )
        except FileNotFoundError:
            self.node.get_logger().info(
                "maze_actors.json not found — analytical collision disabled"
            )
            self.actor_trajectories = []
        except Exception as e:
            self.node.get_logger().warn(f"Actor data load error: {e}")
            self.actor_trajectories = []

    def _compute_actor_positions(self):
        elapsed = time.time() - self.actor_ref_time
        positions = []
        for a in self.actor_trajectories:
            x1, y1 = a["x1"], a["y1"]
            x2, y2 = a["x2"], a["y2"]
            period = a["period"]
            speed = a["speed"]
            dx, dy = x2 - x1, y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-6 or period < 1e-6:
                positions.append((x1, y1))
                continue
            duration = dist / speed
            t_mod = elapsed % period
            if t_mod < duration:
                frac = t_mod / duration
                positions.append((x1 + frac * dx, y1 + frac * dy))
            elif t_mod < duration + 0.5:
                positions.append((x2, y2))
            elif t_mod < 2.0 * duration + 0.5:
                frac = (t_mod - duration - 0.5) / duration
                positions.append((x2 - frac * dx, y2 - frac * dy))
            else:
                positions.append((x1, y1))
        return positions

    def _check_actor_collision(self) -> bool:
        """Dual-layer: YOLO (threat_vector) + analytical trajectory."""
        log = self.node.get_logger()
        hit_source = None

        # Layer 1: YOLO detections
        min_yolo_r = float("inf")
        vec = self.latest_vector
        for i in range(5):
            base = 3 + i * 17
            if base + 16 >= len(vec):
                continue
            is_valid = vec[base + 16]
            if is_valid < 0.5:
                continue
            r_3d = vec[base + 4]
            min_yolo_r = min(min_yolo_r, r_3d)
            if r_3d < self.ACTOR_COLLISION_RADIUS:
                hit_source = f"YOLO r_3d={r_3d:.2f}"

        # Layer 2: analytical trajectory
        min_analytic_d = float("inf")
        if hit_source is None and self.drone_z <= self.ACTOR_COLLISION_Z_MAX:
            for ax, ay in self._compute_actor_positions():
                dx = self.drone_x - ax
                dy = self.drone_y - ay
                d = math.sqrt(dx * dx + dy * dy)
                min_analytic_d = min(min_analytic_d, d)
                if d < self.ACTOR_COLLISION_RADIUS:
                    hit_source = f"ANALYTIC d={d:.2f}"
                    break

        # Periodic status log
        if self.step_count % self.COLLISION_LOG_INTERVAL == 0:
            log.info(
                f"[COL] step={self.step_count} "
                f"drone=({self.drone_x:.1f},{self.drone_y:.1f},z={self.drone_z:.1f}) "
                f"yolo_min={min_yolo_r:.1f} analytic_min={min_analytic_d:.1f} "
                f"actors={len(self.actor_trajectories)}"
            )

        if hit_source:
            log.warn(
                f"COLLISION DETECTED [{hit_source}] "
                f"drone=({self.drone_x:.1f},{self.drone_y:.1f},z={self.drone_z:.1f}) "
                f"step={self.step_count} → episode terminated, resetting"
            )
            return True
        return False

    def _soft_reset_drone(self):
        x, y, z = self.SPAWN_POSITION
        cmd = (
            f"gz service -s /world/{self.GZ_WORLD_NAME}/set_pose "
            f"--reqtype gz.msgs.Pose "
            f"--reptype gz.msgs.Boolean "
            f"--timeout 1000 "
            f"--req 'name: \"{self.DRONE_MODEL_NAME}\", "
            f"position: {{x: {x}, y: {y}, z: {z}}}, "
            f"orientation: {{w: 1.0}}'"
        )
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=3
            )
            if result.returncode != 0:
                self.node.get_logger().warn(
                    f"set_pose failed (rc={result.returncode}): {result.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            self.node.get_logger().warn("set_pose timed out — continuing anyway")
        except Exception as e:
            self.node.get_logger().warn(f"Soft reset error: {e}")

        time.sleep(self.RESET_STABILIZE_SEC)

    def close(self):
        self._running = False
        time.sleep(0.2)
        try:
            self.node.destroy_node()
        except Exception:
            pass