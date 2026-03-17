"""
env.py  –  ThreatAgentEnv  (Asimetrik SAC versiyonu)
─────────────────────────────────────────────────────
Observation space → Dict:
  "obs"        : Box(74,)  – actor görür, deploy'da kullanılır
  "privileged" : Box(43,)  – critic görür, sadece eğitimde

Privileged 43-dim bileşimi:
  ┌──────────┬───────┬──────────────────────────────────────────────────┐
  │ Dilim    │ Boyut │ İçerik                                           │
  ├──────────┼───────┼──────────────────────────────────────────────────┤
  │ [0:5]    │   5   │ target_scores   – reward fn'den gerçek risk      │
  │ [5:15]   │  10   │ actor_token_xy  – token-hizalı [x,y] × 5        │
  │ [15:23]  │   8   │ drone_global    – x,y,z, sin_yaw,cos_yaw,vx,vy,vz│
  │ [23:33]  │  10   │ maze_feat       – cell_face_dist(4)+cell_norm(2) │
  │          │       │                   +walls_cur(4)                   │
  │ [33:38]  │   5   │ waypoint_rel    – dx,dy,dist,sin_angle,cos_angle │
  │ [38:43]  │   5   │ prev_action                                      │
  └──────────┴───────┴──────────────────────────────────────────────────┘
  Toplam: 5+10+8+10+5+5 = 43

Düzeltmeler (önceki sürüme göre):
  1. actor_token_xy : obs'daki _sort sonrası token sırası ile hizalı;
                      priv[5+2i : 5+2i+2] == token i'nin dünya (x,y)'si
  2. yaw → sin_yaw, cos_yaw : tek scalar'ın [-π,π] süreksizliği giderildi
  3. cell_face_dist : "wall_dist" adı değişti; bu değer gerçek duvar
                      mesafesi değil, hücre yüzüne olan normalize mesafe
  4. waypoint_rel   : yalnızca cos_angle yerine (sin_angle, cos_angle)
                      kullanılıyor; yön tam olarak kodlanıyor

ROS subscriptions:
  /threat/state_vec          Float32MultiArray  (obs 74-dim)
  /bev/image                 Image
  /curriculum/actor_poses    PoseArray          (ham aktör konumları)
  /drone/odom                nav_msgs/Odometry  (global pose + vel)
  /drone/waypoint            geometry_msgs/PointStamped
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, PointStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


OBS_DIM   = 74
PRIV_DIM  = 43
K         = 5
TOKEN_LEN = 7

MAZE_ROWS       = 15
MAZE_COLS       = 15
MAZE_CELL_SIZE  = 3.0
DRONE_SPAWN_CELL = (MAZE_ROWS // 2, MAZE_COLS // 2)
WALLS_PATH      = "/home/ubuntu/Desktop/maze_walls.json"

TOPIC_STATE_VEC   = "/threat/state_vec"
TOPIC_BEV         = "/bev/image"
TOPIC_ACTOR_POSES = "/curriculum/actor_poses"
TOPIC_DRONE_ODOM  = "/odometry/filtered"
TOPIC_WAYPOINT    = "/plan"

WAYPOINT_MAX_DIST = 50.0   


class MazeContextExtractor:
    def __init__(
        self,
        walls_path: str            = WALLS_PATH,
        rows:       int            = MAZE_ROWS,
        cols:       int            = MAZE_COLS,
        cell_size:  float          = MAZE_CELL_SIZE,
        spawn_cell: Tuple[int,int] = DRONE_SPAWN_CELL,
    ):
        self.rows      = rows
        self.cols      = cols
        self.cell_size = cell_size
        sr, sc         = spawn_cell
        self.ox        = -(sc + 0.5) * cell_size
        self.oy        = -(sr + 0.5) * cell_size
        self.walls: Optional[List[List[Dict[str,bool]]]] = None
        self._load(walls_path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            print(f"[MazeContext] Dosya bulunamadı: {path}")
            return
        try:
            with open(path) as f:
                self.walls = json.load(f)
            print(f"[MazeContext] Yüklendi: {path}  "
                  f"({len(self.walls)}×{len(self.walls[0])} hücre)")
        except Exception as e:
            print(f"[MazeContext] Hata: {e}")

    def world_to_cell(self, wx: float, wy: float) -> Tuple[int, int]:
        col = int((wx - self.ox) / self.cell_size)
        row = int((wy - self.oy) / self.cell_size)
        return (max(0, min(self.rows - 1, row)),
                max(0, min(self.cols - 1, col)))

    def get_features(self, wx: float, wy: float) -> np.ndarray:
        feat = np.zeros(10, dtype=np.float32)
        if self.walls is None:
            return feat

        row, col = self.world_to_cell(wx, wy)
        cs       = self.cell_size


        rel_x = float(np.clip(wx - (self.ox + col * cs), 0.0, cs))
        rel_y = float(np.clip(wy - (self.oy + row * cs), 0.0, cs))

        feat[0] = rel_y / cs          
        feat[1] = (cs - rel_x) / cs  
        feat[2] = (cs - rel_y) / cs  
        feat[3] = rel_x / cs          

      
        feat[4] = row / max(self.rows - 1, 1)
        feat[5] = col / max(self.cols - 1, 1)


        cw       = self.walls[row][col]
        feat[6]  = 1.0 if cw.get("N", True) else 0.0
        feat[7]  = 1.0 if cw.get("E", True) else 0.0
        feat[8]  = 1.0 if cw.get("S", True) else 0.0
        feat[9]  = 1.0 if cw.get("W", True) else 0.0

        return feat


class ThreatAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(
                low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
            ),
            "privileged": spaces.Box(
                low=-np.inf, high=np.inf, shape=(PRIV_DIM,), dtype=np.float32
            ),
        })
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(K,), dtype=np.float32
        )

        self.K         = K
        self.token_len = TOKEN_LEN
        self.lidar_sectors = 36

        self.prev_action       = np.zeros(K, dtype=np.float32)
        self.class_seen_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.shuffle_prob      = 0.5

        self.curriculum_stage  = 1
        self.enable_curriculum = True

        self.max_episode_steps = 2048
        self.episode_idx       = 0
        self.episode_step      = 0
        self.global_step       = 0

    
        self.maze_ctx = MazeContextExtractor()
        self._drone_global = np.zeros(8, dtype=np.float32)
        self._drone_global[4] = 1.0   


        self._waypoint_xy: Optional[Tuple[float, float]] = None

        self._raw_actor_positions: List[Tuple[float, float]] = []
        self._latest_target_scores = np.zeros(K, dtype=np.float32)

        self.save_bev_gifs       = True
        self.frame_save_interval = 32
        self.keep_png_frames     = False
        self.base_frame_dir = "/home/ubuntu/Desktop/ros2_env/bev_episode_frames"
        self.base_gif_dir   = "/home/ubuntu/Desktop/ros2_env/bev_episode_gifs"
        os.makedirs(self.base_frame_dir, exist_ok=True)
        os.makedirs(self.base_gif_dir,   exist_ok=True)

        if not rclpy.ok():
            rclpy.init()

        self.node_name = f"gym_threat_{str(uuid.uuid4())[:8]}"
        self.node      = rclpy.create_node(self.node_name)
        self._running  = True

        self.latest_vector = np.zeros(OBS_DIM, dtype=np.float32)
        self.new_vec       = False
        self.cond          = threading.Condition()

        self.bridge     = CvBridge()
        self.latest_bev = None

        self.sub_vec    = self.node.create_subscription(
            Float32MultiArray, TOPIC_STATE_VEC, self._vec_callback, 10)
        self.sub_img    = self.node.create_subscription(
            Image, TOPIC_BEV, self._img_callback, 10)
        self.sub_actors = self.node.create_subscription(
            PoseArray, TOPIC_ACTOR_POSES, self._actor_poses_callback, 10)
        self.sub_odom   = self.node.create_subscription(
            Odometry, TOPIC_DRONE_ODOM, self._odom_callback, 10)
        self.sub_wp     = self.node.create_subscription(
            PointStamped, TOPIC_WAYPOINT, self._waypoint_callback, 10)

        self._spin_thread = threading.Thread(target=self._spin_node, daemon=True)
        self._spin_thread.start()
        time.sleep(1.0)

    def _odom_callback(self, msg: Odometry) -> None:
        p  = msg.pose.pose.position
        q  = msg.pose.pose.orientation
        t  = msg.twist.twist.linear

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        norm      = math.sqrt(siny_cosp**2 + cosy_cosp**2) + 1e-9

        self._drone_global = np.array([
            p.x, p.y, p.z,
            siny_cosp / norm, 
            cosy_cosp / norm, 
            t.x, t.y, t.z,
        ], dtype=np.float32)

    def _waypoint_callback(self, msg: PointStamped) -> None:
        self._waypoint_xy = (float(msg.point.x), float(msg.point.y))

    def _actor_poses_callback(self, msg: PoseArray) -> None:
        self._raw_actor_positions = [
            (float(p.position.x), float(p.position.y))
            for p in msg.poses
        ]

    def _img_callback(self, msg: Image) -> None:
        try:
            self.latest_bev = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8")
        except Exception as e:
            self.node.get_logger().warn(f"img_callback: {e}")

    def _vec_callback(self, msg: Float32MultiArray) -> None:
        try:
            data = np.array(msg.data, dtype=np.float32)
            if data.shape[0] != OBS_DIM:
                return
            data = self._filter_stale(data)
            data = self._apply_curriculum_mask(data)
            data = self._sort_objects(data)
            with self.cond:
                self.latest_vector = data
                self.new_vec       = True
                self.cond.notify_all()
        except Exception as e:
            self.node.get_logger().warn(f"vec_callback: {e}")

    def _spin_node(self) -> None:
        exc = rclpy.executors.SingleThreadedExecutor()
        exc.add_node(self.node)
        try:
            while self._running and rclpy.ok():
                exc.spin_once(timeout_sec=0.05)
        except Exception as e:
            self.node.get_logger().warn(f"spin: {e}")
        finally:
            exc.remove_node(self.node)


    def _build_privileged(
        self,
        target_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ts = (np.clip(target_scores, 0.0, 1.0).astype(np.float32)
              if target_scores is not None
              else self._latest_target_scores.copy())
        if len(ts) < K:
            ts = np.pad(ts, (0, K - len(ts)))
        ts = ts[:K]


        actor_xy = self._align_actors_to_tokens()   
        drone_g = self._drone_global.copy()

        maze_f = self.maze_ctx.get_features(
            float(drone_g[0]), float(drone_g[1])
        )


        wp = self._calc_waypoint_rel(
            float(drone_g[0]), float(drone_g[1]),
            float(drone_g[3]), float(drone_g[4])   
        )

        prev_a = self.prev_action.copy()

        priv = np.concatenate([ts, actor_xy, drone_g, maze_f, wp, prev_a])
        assert priv.shape == (PRIV_DIM,), \
            f"PRIV_DIM uyuşmuyor: {priv.shape[0]} != {PRIV_DIM}"
        return priv

    def _align_actors_to_tokens(self) -> np.ndarray:
        result = np.zeros((K, 2), dtype=np.float32)

        if not self._raw_actor_positions:
            return result.flatten()

        actors      = np.array(self._raw_actor_positions, dtype=np.float32)  
        objs        = self.latest_vector[39:].reshape(K, TOKEN_LEN)
        drone_xy    = np.array(
            [float(self._drone_global[0]), float(self._drone_global[1])],
            dtype=np.float32
        )

        actor_dists = np.linalg.norm(actors - drone_xy, axis=1)  

   
        valid_token_idx = [i for i in range(K) if objs[i, 6] > 0.5]
        if not valid_token_idx:
            return result.flatten()

        n_tok   = len(valid_token_idx)
        n_act   = len(actors)

        tok_dists = np.array(
            [float(objs[ti, 1]) for ti in valid_token_idx], dtype=np.float32
        )                                                           
        cost = np.abs(
            tok_dists[:, np.newaxis] - actor_dists[np.newaxis, :]
        )                                                           

        pairs = sorted(
            ((cost[ti_pos, ai], ti_pos, ai)
             for ti_pos in range(n_tok)
             for ai     in range(n_act)),
            key=lambda x: x[0]
        )

        assigned_tok_pos = set()
        assigned_act_idx = set()
        for c, ti_pos, ai in pairs:
            if ti_pos in assigned_tok_pos or ai in assigned_act_idx:
                continue
            result[valid_token_idx[ti_pos]] = actors[ai]
            assigned_tok_pos.add(ti_pos)
            assigned_act_idx.add(ai)
            if len(assigned_tok_pos) == min(n_tok, n_act):
                break

        return result.flatten()  

    def _calc_waypoint_rel(
        self,
        drone_x: float, drone_y: float,
        sin_yaw: float, cos_yaw: float,
    ) -> np.ndarray:
    
        if self._waypoint_xy is None:
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        wx, wy = self._waypoint_xy
        ddx    = wx - drone_x
        ddy    = wy - drone_y
        dist   = math.sqrt(ddx**2 + ddy**2)

        norm_d = WAYPOINT_MAX_DIST

        wp_angle = math.atan2(ddy, ddx)
        drone_yaw = math.atan2(sin_yaw, cos_yaw)

        angle_diff = wp_angle - drone_yaw

        return np.array([
            float(np.clip(ddx / norm_d, -1.0, 1.0)),
            float(np.clip(ddy / norm_d, -1.0, 1.0)),
            float(np.clip(dist / norm_d, 0.0, 1.0)),
            float(math.sin(angle_diff)),
            float(math.cos(angle_diff)),
        ], dtype=np.float32)

    def _get_obs_dict(
        self,
        target_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        return {
            "obs":        self.latest_vector.copy(),
            "privileged": self._build_privileged(target_scores),
        }


    def _sort_objects(self, v: np.ndarray) -> np.ndarray:
        try:
            objs = v[39:].reshape(K, TOKEN_LEN)
            vi   = np.where(objs[:, 6] > 0.5)[0]
            ii   = np.where(objs[:, 6] <= 0.5)[0]
            if np.random.rand() < self.shuffle_prob:
                np.random.shuffle(vi)
                idx = np.concatenate([vi, ii])
            else:
                idx = np.concatenate([vi[np.argsort(objs[vi, 1])], ii])
            v[39:] = objs[idx].flatten()
        except Exception:
            pass
        return v

    def _apply_curriculum_mask(self, v: np.ndarray) -> np.ndarray:
        if not self.enable_curriculum or self.curriculum_stage == 3:
            return v
        objs = v[39:].reshape(K, TOKEN_LEN).copy()
        for i in range(K):
            cls_id = int(objs[i, 0])
            if objs[i, 6] > 0.5:
                if self.curriculum_stage == 1 and cls_id != 4:
                    objs[i, 6] = 0.0
                elif self.curriculum_stage == 2 and cls_id not in [0, 4]:
                    objs[i, 6] = 0.0
        v[39:] = objs.flatten()
        return v

    def _filter_stale(self, v: np.ndarray) -> np.ndarray:
        objs = v[39:].reshape(K, TOKEN_LEN).copy()
        for i in range(K):
            if objs[i, 6] > 0.5 and objs[i, 1] > 15.0 and abs(objs[i, 2]) < 0.2:
                objs[i, 6] = 0.0
        v[39:] = objs.flatten()
        return v

    def _wait_for_obs(self, timeout: float = 0.5) -> bool:
        end = time.time() + timeout
        with self.cond:
            while time.time() < end:
                if self.new_vec:
                    self.new_vec = False
                    return True
                rem = end - time.time()
                if rem > 0:
                    self.cond.wait(timeout=rem)
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_action           = np.zeros(K, dtype=np.float32)
        self._latest_target_scores = np.zeros(K, dtype=np.float32)
        self.episode_step          = 0
        self.class_seen_counts     = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self._wait_for_obs(timeout=1.0)
        return self._get_obs_dict(), {}

    def step(self, action: np.ndarray):
        self._wait_for_obs(timeout=0.2)
        obs_vec          = self.latest_vector.copy()
        reward, info     = self.calculate_reward_and_info(action, obs_vec)

        ts = np.array(
            [t["TRGT"] for t in info["top_threats"]], dtype=np.float32
        ) if info["top_threats"] else np.zeros(K, dtype=np.float32)
        if len(ts) < K:
            ts = np.pad(ts, (0, K - len(ts)))
        self._latest_target_scores = ts[:K]

        obs_dict = self._get_obs_dict(self._latest_target_scores)

        self._save_bev_frame(reward=reward, metrics=info["metrics"])
        self.prev_action   = action.copy()
        self.global_step  += 1
        self.episode_step += 1

        terminated = False
        truncated  = self.episode_step >= self.max_episode_steps
        if truncated:
            self._make_episode_gif()
            self.episode_idx += 1

        return obs_dict, reward, terminated, truncated, info


    def calculate_reward_and_info(
        self, action: np.ndarray, obs: np.ndarray
    ) -> Tuple[float, Dict]:
        cls_names    = {0: "Unknown", 1: "Drone", 2: "Bird",
                        3: "FixedWing", 4: "Person"}
        objects_flat = obs[39:]
        total_reward = 0.0
        valid_count  = 0
        threats      = []
        all_tgt      = []
        all_act      = []

        for i in range(K):
            obj       = objects_flat[i * TOKEN_LEN : (i + 1) * TOKEN_LEN]
            cls_id    = int(obj[0])
            dist      = obj[1]
            cspeed    = obj[2]
            conf      = obj[5]
            is_valid  = obj[6]
            score     = float(action[i])

            if cls_id not in cls_names:
                is_valid = 0.0

            if is_valid > 0.5:
                valid_count += 1
                if cls_id in self.class_seen_counts:
                    self.class_seen_counts[cls_id] += 1

                d_score  = (1.0 / (1.0 + np.exp(2.0 * (dist - 3.5)))
                            if cls_id == 4
                            else 1.0 / (1.0 + np.exp(1.5 * (dist - 2.5))))
                sp_score = np.clip(0.3 * cspeed, 0.0, 0.8) if cspeed > 0.1 else 0.0
                raw_risk = np.clip(d_score + sp_score, 0.0, 1.0)

                if cls_id == 0:
                    cf = 0.8 if cspeed > 0.3 else 0.4 if cspeed > 0.1 else 0.05
                elif cls_id == 4:
                    cf = 0.9
                else:
                    cf = 0.0

                tgt = raw_risk * cf
                all_tgt.append(tgt)
                all_act.append(score)

                threats.append({
                    "id":    i,
                    "cls":   cls_names.get(cls_id, "?"),
                    "dist":  round(float(dist),  1),
                    "vel":   round(float(cspeed), 1),
                    "conf":  round(float(conf),   2),
                    "score": round(score,          2),
                    "TRGT":  round(float(tgt),     2),
                })

                diff = abs(score - tgt)
                alr  = 1.0 - 2.0 * diff
                if tgt > 0.5 and score > 0.5: alr += 0.2
                if tgt > 0.5 and score < 0.5: alr -= 0.3
                total_reward += alr
            else:
                total_reward -= score
                all_tgt.append(0.0)
                all_act.append(score)

        total_reward = float(np.clip(total_reward, -5.0, 5.0))
        arr_a = np.array(all_act)
        arr_t = np.array(all_tgt)
        hm    = arr_t > 0.4
        cov   = (float(np.sum(arr_a[hm] > 0.6) / np.sum(hm))
                 if np.any(hm) else 0.0)

        return total_reward, {
            "top_threats": threats,
            "metrics": {
                "valid_obj_count":    valid_count,
                "high_risk_coverage": cov,
                "mean_score_var":     float(np.var(arr_a)),
                "temporal_jitter":    float(np.mean(np.abs(arr_a - self.prev_action))),
                "drone_seen_count":   self.class_seen_counts[1],
                "person_seen_count":  self.class_seen_counts[4],
            },
            "total_reward": total_reward,
        }

    def set_curriculum_stage(self, stage: int) -> None:
        self.curriculum_stage = int(np.clip(stage, 1, 3))

    def reload_maze(self, walls_path: str = WALLS_PATH) -> None:
        self.maze_ctx._load(walls_path)

    def _get_episode_frame_dir(self) -> str:
        d = os.path.join(self.base_frame_dir, f"episode_{self.episode_idx:03d}")
        os.makedirs(d, exist_ok=True)
        return d

    def _save_bev_frame(self, reward=None, metrics=None) -> None:
        if not self.save_bev_gifs or self.latest_bev is None:
            return
        if self.episode_step % self.frame_save_interval != 0:
            return
        vis = self.latest_bev.copy()
        cv2.putText(
            vis,
            f"Ep:{self.episode_idx} St:{self.episode_step} S:{self.curriculum_stage}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        if reward is not None:
            cv2.putText(vis, f"R:{reward:.3f}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if metrics:
            y = 75
            for k, v in list(metrics.items())[:4]:
                label = f"{k}:{v:.2f}" if isinstance(v, float) else f"{k}:{v}"
                cv2.putText(vis, label, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y += 22
        cv2.imwrite(
            os.path.join(self._get_episode_frame_dir(),
                         f"frame_{self.episode_step:04d}.png"),
            vis
        )

    def _make_episode_gif(self) -> None:
        ep_dir = self._get_episode_frame_dir()
        files  = sorted(f for f in os.listdir(ep_dir) if f.endswith(".png"))
        if not files:
            return
        images = [imageio.imread(os.path.join(ep_dir, f)) for f in files]
        gif_path = os.path.join(
            self.base_gif_dir, f"episode_{self.episode_idx:03d}.gif"
        )
        imageio.mimsave(gif_path, images, duration=0.12)
        if not self.keep_png_frames:
            for f in files:
                os.remove(os.path.join(ep_dir, f))
            os.rmdir(ep_dir)


    def close(self) -> None:
        self._running = False
        time.sleep(0.2)
        try:
            self.node.destroy_node()
        except Exception:
            pass