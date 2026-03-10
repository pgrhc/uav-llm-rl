import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import threading
import time
import uuid
import os
import cv2
import imageio.v2 as imageio
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class ThreatAgentEnv(gym.Env):
    def __init__(self):
        super(ThreatAgentEnv, self).__init__()
        
        self.K = 5
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.K,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(74,),  
            dtype=np.float32
        )

        self.token_len = 7  
        self.lidar_sectors = 36

        self.prev_action = np.zeros(self.K, dtype=np.float32)
        self.class_seen_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.shuffle_prob = 0.5
        

        self.curriculum_stage = 1
        self.enable_curriculum = True


        if not rclpy.ok():
            rclpy.init()
            
        self.node_name = f'gym_threat_{str(uuid.uuid4())[:8]}'
        self.node = rclpy.create_node(self.node_name)
        self.node.get_logger().info(f"ThreatAgentEnv V2 başlatıldı: {self.node_name}")
        self._running = True
        self.latest_vector = np.zeros((74,), dtype=np.float32)  
        self.new_vec = False
        self.cond = threading.Condition()
        self.sub_vec = self.node.create_subscription(
            Float32MultiArray, '/threat/state_vec', self.vec_callback, 10)
        self.sub_img = self.node.create_subscription(
            Image,
            '/bev/image',
            self.img_callback,
            10
        )
           
        self.max_episode_steps = 2048
        self.episode_idx = 0
        self.episode_step = 0
        self.global_step = 0

        self.save_bev_gifs = True
        self.frame_save_interval = 32
        self.keep_png_frames = False

        self.base_frame_dir = "/home/ubuntu/Desktop/ros2_env/bev_episode_frames"
        self.base_gif_dir = "/home/ubuntu/Desktop/ros2_env/bev_episode_gifs"
        os.makedirs(self.base_frame_dir, exist_ok=True)
        os.makedirs(self.base_gif_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.latest_bev = None
        self.new_img = False

        self.thread = threading.Thread(target=self._spin_node, daemon=True)
        self.thread.start()
        time.sleep(1.0)


    def img_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_bev = cv_img
            self.new_img = True
        except Exception as e:
            self.node.get_logger().warn(f"img_callback hatası: {e}")

    def _get_episode_frame_dir(self):
        ep_dir = os.path.join(self.base_frame_dir, f"episode_{self.episode_idx:03d}")
        os.makedirs(ep_dir, exist_ok=True)
        return ep_dir
    
    def _save_bev_frame(self, reward=None, metrics=None):
        if not self.save_bev_gifs:
            return

        if self.latest_bev is None:
            return

        if self.episode_step % self.frame_save_interval != 0:
            return

        vis = self.latest_bev.copy()

        cv2.putText(vis, f"Episode: {self.episode_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Ep step: {self.episode_step}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Global step: {self.global_step}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if reward is not None:
            cv2.putText(vis, f"Reward: {reward:.3f}", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y = 145

        if metrics is not None:

            cv2.putText(vis, f"valid_obj: {metrics['valid_obj_count']}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y += 25

            cv2.putText(vis, f"risk_cov: {metrics['high_risk_coverage']:.2f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y += 25

            cv2.putText(vis, f"score_var: {metrics['mean_score_var']:.3f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y += 25

            cv2.putText(vis, f"jitter: {metrics['temporal_jitter']:.3f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y += 25

            cv2.putText(vis, f"drone_seen: {metrics['drone_seen_count']}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y += 25

            cv2.putText(vis, f"person_seen: {metrics['person_seen_count']}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        ep_dir = self._get_episode_frame_dir()
        frame_path = os.path.join(ep_dir, f"frame_{self.episode_step:04d}.png")
        cv2.imwrite(frame_path, vis)

    def _make_episode_gif(self):
        ep_dir = self._get_episode_frame_dir()
        files = sorted([f for f in os.listdir(ep_dir) if f.endswith(".png")])

        if not files:
            self.node.get_logger().warn(f"Episode {self.episode_idx} için frame bulunamadı.")
            return

        images = []
        for f in files:
            img_path = os.path.join(ep_dir, f)
            images.append(imageio.imread(img_path))

        gif_path = os.path.join(
            self.base_gif_dir,
            f"episode_{self.episode_idx:03d}.gif"
        )

        imageio.mimsave(gif_path, images, duration=0.12)
        self.node.get_logger().info(f"GIF kaydedildi: {gif_path}")

        if not self.keep_png_frames:
            for f in files:
                os.remove(os.path.join(ep_dir, f))
            os.rmdir(ep_dir)

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
            lidar_data = vector[3:39]
            objs = vector[39:].reshape(5, 7)

            valid = objs[:, 6] > 0.5
            valid_idx = np.where(valid)[0]
            invalid_idx = np.where(~valid)[0]

            if np.random.rand() < self.shuffle_prob:
                np.random.shuffle(valid_idx)              
                sorted_idx = np.concatenate([valid_idx, invalid_idx])
            else:
                d = objs[valid_idx, 1]                    
                valid_sorted = valid_idx[np.argsort(d)]
                sorted_idx = np.concatenate([valid_sorted, invalid_idx])

            sorted_objs = objs[sorted_idx]
            return np.concatenate((uav_data, lidar_data, sorted_objs.flatten()))
        except Exception:
            return vector

    def _apply_curriculum_mask(self, vector):
        if not self.enable_curriculum or self.curriculum_stage == 3:
            return vector
        
        objects_flat = vector[39:].reshape(5, 7).copy()  
        
        for i in range(5):
            class_id = int(objects_flat[i][0]) 
            is_valid = objects_flat[i][6]      
            
            if is_valid > 0.5:
                if self.curriculum_stage == 1:
                    if class_id != 4:
                        objects_flat[i][6] = 0.0
                elif self.curriculum_stage == 2:
                    if class_id not in [0, 4]:
                        objects_flat[i][6] = 0.0
        
        vector[39:] = objects_flat.flatten()
        return vector

    def vec_callback(self, msg):
        try:
            data = np.array(msg.data, dtype=np.float32)
            if data.shape[0] == 74: 
                data = self._filter_stale_objects(data)
                sorted_data = self._sort_objects(data)
                masked_data = self._apply_curriculum_mask(sorted_data)
                with self.cond:
                    self.latest_vector = masked_data
                    self.new_vec = True
                    self.cond.notify_all()
        except Exception as e:
            self.node.get_logger().warn(f"vec_callback hatası: {e}")

    def _filter_stale_objects(self, vector):
        """
        Stale/phantom objeleri temizle.
        """
        objects_flat = vector[39:].reshape(5, 7).copy() 
        
        for i in range(5):
            dist = objects_flat[i][1]           
            closing_speed = abs(objects_flat[i][2])  
            is_valid = objects_flat[i][6]       
            
            if is_valid > 0.5:
                if dist > 15.0 and closing_speed < 0.2:
                    objects_flat[i][6] = 0.0
                    
        vector[39:] = objects_flat.flatten()
        return vector

    def _wait_for_obs(self, timeout=0.5):
        end = time.time() + timeout
        with self.cond:
            while time.time() < end:
                if self.new_vec:
                    self.new_vec = False
                    return True
                remaining = end - time.time()
                if remaining > 0:
                    self.cond.wait(timeout=remaining)
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_action = np.zeros(self.K, dtype=np.float32)
        self.episode_step = 0
        self._wait_for_obs(timeout=1.0)
        observation = self.latest_vector.copy()
        
        return observation, {}

    def step(self, action):
        self._wait_for_obs(timeout=0.2)
        
        obs = self.latest_vector.copy()
        reward, info_data = self.calculate_reward_and_info(action, obs)
        
        self._save_bev_frame(reward=reward,metrics=info_data["metrics"])
        self.prev_action = action.copy()
        self.global_step += 1
        self.episode_step += 1

        terminated = False 
        truncated = False 
        if self.episode_step >= self.max_episode_steps:
            truncated = True
            self._make_episode_gif()
            self.episode_idx += 1
        
        return obs, reward, terminated, truncated, info_data

    def calculate_reward_and_info(self, action, obs):
        total_reward = 0.0
        vector = obs
        objects_flat = vector[39:]
        
        valid_obj_count = 0
        detailed_threats = []
        
        class_names = {0: "Unknown", 1: "Drone", 2: "Bird", 3: "FixedWing", 4: "Person"}
        
        all_target_risks = []
        all_actions = []

        for i in range(self.K):
            start_idx = i * self.token_len
            obj_data = objects_flat[start_idx : start_idx + self.token_len]
            class_id = int(obj_data[0])     
            dist = obj_data[1]              
            closing_speed = obj_data[2]     
            confidence = obj_data[5]        
            is_valid = obj_data[6]           

            if class_id not in [0, 1, 2, 3, 4]:
                is_valid = 0.0

            current_score = float(action[i])
            
            if is_valid > 0.5:
                valid_obj_count += 1
                
                
                if class_id in self.class_seen_counts:
                    self.class_seen_counts[class_id] += 1
                
                if class_id == 4: 
                    dist_score = 1.0 / (1.0 + np.exp(2.0 * (dist - 3.5)))
                else:
                    dist_score = 1.0 / (1.0 + np.exp(1.5 * (dist - 2.5)))
                
              
                speed_score = 0.0
                if closing_speed > 0.1:
                    speed_score = np.clip(0.3 * closing_speed, 0.0, 0.8)
                
                raw_risk = np.clip(dist_score + speed_score, 0.0, 1.0)
                
           
                if class_id == 0:  
                    if closing_speed > 0.3:
                        c_factor = 0.8
                    elif closing_speed > 0.1:
                        c_factor = 0.4
                    else:
                        c_factor = 0.05
                elif class_id == 4: 
                    c_factor = 0.9
                else: 
                    c_factor = 0.0
                
                target_risk = raw_risk * c_factor
                
                all_target_risks.append(target_risk)
                all_actions.append(current_score)
                
                
                threat_info = {
                    "id": i,  
                    "cls": class_names.get(class_id, "?"),
                    "dist": round(float(dist), 1),
                    "vel": round(float(closing_speed), 1),
                    "conf": round(float(confidence), 2),
                    "score": round(current_score, 2),
                    "TRGT": round(float(target_risk), 2)
                }
                detailed_threats.append(threat_info)
                
                diff = abs(current_score - target_risk)
                alignment_reward = 1.0 - 2.0 * diff
                if target_risk > 0.5 and current_score > 0.5:
                    alignment_reward += 0.2

                if target_risk > 0.5 and current_score < 0.5:
                    alignment_reward -= 0.3
                total_reward += alignment_reward
                
            else:
                ghost_penalty = current_score
                total_reward -= ghost_penalty
                
                all_target_risks.append(0.0)
                all_actions.append(current_score)

        total_reward = np.clip(total_reward, -5.0, 5.0)
        arr_actions = np.array(all_actions)
        arr_targets = np.array(all_target_risks)
        
        metrics = {
            "mean_score_var": float(np.var(arr_actions)),
            "temporal_jitter": float(np.mean(np.abs(arr_actions - self.prev_action))),
            "drone_seen_count": self.class_seen_counts[1],
            "bird_seen_count": self.class_seen_counts[2],
            "person_seen_count": self.class_seen_counts[4],
            "valid_obj_count": valid_obj_count
        }
        
      
        high_risk_mask = arr_targets > 0.4
        if np.any(high_risk_mask):
            coverage = np.sum(arr_actions[high_risk_mask] > 0.6) / np.sum(high_risk_mask)
            metrics["high_risk_coverage"] = float(coverage)
        else:
            metrics["high_risk_coverage"] = 0.0

        info_data = {
            "top_threats": detailed_threats,
            "metrics": metrics,
            "total_reward": float(total_reward)
        }
            
        return total_reward, info_data
    
    def set_curriculum_stage(self, stage):
        self.curriculum_stage = np.clip(stage, 1, 3)
        self.node.get_logger().info(f"Curriculum stage: {self.curriculum_stage}")
    
    def enable_curriculum_learning(self, enable=True):
        self.enable_curriculum = enable
        self.node.get_logger().info(f"Curriculum: {'ACTIVE' if enable else 'DISABLED'}")

    def close(self):
        self._running = False
        time.sleep(0.2)
        try:
            self.node.destroy_node()
        except Exception:
            pass