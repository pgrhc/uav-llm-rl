#!/usr/bin/env python3
"""
Route Planning Agent — Curriculum Learning Environment

Observation (same as RouteEnv):
    costmap_patch  : (1, 64, 64)  — CNN /route/costmap_patch (node: harita eksenli; isteğe bağlı body yaw döndürme)
    threat_vector  : (74,)        — MLP   /threat/state_vec
    threat_scores  : (5,)         — MLP   /threat/target_scores
    goal_state     : (7,)         — MLP   (rel_goal, dist, speed, yaw)
    a_star_path    : (10,)        — MLP   /plan  (5 wp × rel_x,rel_y)

Action:
    Box(4,) → (dx, dy, dz, dyaw) body frame, scaled by STEP_SIZE / Z_STEP / MAX_YAW_RATE
    ROUTE_STEP_SIZE (env, m) — xy adım ölçeği, varsayılan 0.3 (ör. 0.2 daha yumuşak waypoint)

    Hybrid A* baseline (default, ROUTE_HYBRID_ASTAR=1):
        waypoint_xy = lookahead + threat_gate × body-frame residual (dx,dy)
        wp_z        = CRUISE_Z + threat_gate × dz (clip ±ROUTE_HYBRID_Z_CLIP) veya ROUTE_HYBRID_USE_Z_ACTION=0 iken sabit CRUISE_Z
        yaw         = hedefe doğru + threat_gate × küçük dyaw düzeltmesi
        threat_gate = sigmoid(effective_max; ROUTE_THREAT_GATE_THRESHOLD, ROUTE_THREAT_GATE_K)
          effective_max = max(target_scores) [+ Stage1 sentetik gürültü isteğe bağlı]
    ROUTE_HYBRID_ASTAR=0: waypoint = drone + threat_gate × offset (tam 4D)
    ROUTE_THREAT_GATE=0: gate=1.0; path cezası sabit 0.5×path_err_norm (kapı ölçeği yok)

Reward (step() ile birebir — VecNormalize öncesi ham r):
    r_progress          +3.0 × (prev_dist_to_goal − dist)   (hedef yoksa 0)
    r_goal              +50.0  (0.5 m içinde, çarpışma yoksa)
    r_collision         −100.0 (öncelikli; costmap wp+drone / LiDAR / aktör)
    r_path              −0.5 × min(path_err/5,1) × path_w
          path_w = PATH_ERR_SCALE_LOW×(1−g)+PATH_ERR_SCALE_HIGH×g  (THREAT_GATE açıkken)
          değilse path_w = 1.0
    r_astar_return      +ASTAR_RETURN_BONUS (kapı açık, g küçük, path_err küçükse)
    r_path_drift        −50.0  (path_err > PATH_ERROR_TERMINATE_M, plan yeterliyse)
    r_threat            −2.0 × max(threat_scores)  (adım sonrası güncel skorlar)
    r_smooth            −0.05 × ‖a − a_prev‖
    r_time              −0.1

    info: rw_* alanlarında bu bileşenler ayrı ayrı (log/TensorBoard için).

Stages:
    1  Path following — open maze, no actors
    2  Static obstacles — narrow corridors, dead-ends
    3  Dynamic threats — 180-200 walking actors

Eğitim (SAC) ile:
    train script SB3 timestep’i set_sb3_timesteps ile yazar. ROUTE_RANDOM_PHASE_ACTION_SCALE<1
    ve ROUTE_LEARNING_STARTS ile eşleşen süre boyunca aksiyon bu faktörle küçültülür (random toplama fazı).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import threading
import time
import json
import os
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2

# Curriculum maze sabitleri (maze_curriculum_world.py ile ayni degerler)
_WALLS_DIR = "/home/ubuntu/Desktop"
STAGE_ORIGINS = {1: (0.0, 0.0), 2: (1000.0, 0.0), 3: (2000.0, 0.0)}
WALLS_PATHS = {
    1: os.path.join(_WALLS_DIR, "maze_walls_stage1.json"),
    2: os.path.join(_WALLS_DIR, "maze_walls_stage2.json"),
    3: os.path.join(_WALLS_DIR, "maze_walls_stage3.json"),
}
ACTORS_JSON_PATH = os.path.join(_WALLS_DIR, "maze_actors_stage3.json")
ACTORS_UNIFIED_JSON_PATH = os.path.join(_WALLS_DIR, "maze_actors_unified.json")
WALLS_UNIFIED_PATH = os.path.join(_WALLS_DIR, "maze_walls_unified.json")

# Birlesik maze: tek maze, pozisyon bazli stage, teleport yok
UNIFIED_MAZE = True  # True = birlesik maze, False = 3 ayri maze
UNIFIED_ORIGIN = (0.0, 0.0)
# Stage boundaries (must match maze_curriculum_world.py layout)
SECTION1_X_MAX = 50.0
SECTION2_X_MAX = 125.0

# Asymmetric SAC: privileged obs dim (critic only)
PRIV_DIM = 102  # global_emb(64) + actor_rel(10) + collision(4) + path(20) + prev_action(4)


class RouteCurriculumEnv(gym.Env):
    metadata = {"render_modes": []}

    STEP_SIZE = float(os.environ.get("ROUTE_STEP_SIZE", "0.3"))
    Z_STEP = 0.2
    MAX_YAW_RATE = 0.52
    GOAL_TOLERANCE = 0.5
    SAFETY_RADIUS = 0.5
    LETHAL_THRESHOLD = 90
    MAX_EPISODE_STEPS = 1000
    MAX_GOAL_DIST = 30.0
    MAX_SPEED = 5.0

    LIDAR_MAX_RANGE = 30.0
    LIDAR_COLLISION_M = 0.4
    LIDAR_START_IDX = 3
    LIDAR_END_IDX = 39
    NUM_PATH_WPS = 5

    DRONE_MODEL_NAME = "x500_mono_cam_0"
    GZ_WORLD_NAME = "default"
    RESET_STABILIZE_SEC = 2.0  # 5.0 → 2.0

    # Hibrit rota: /plan üzerindeki lookahead + SAC rezidüeli; 0=eski offset modu
    HYBRID_ASTAR_BASELINE = os.environ.get("ROUTE_HYBRID_ASTAR", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    # Yayınlanan waypoint irtifası (dz aksiyonu hybrid modda kullanılmaz)
    CRUISE_Z = float(os.environ.get("ROUTE_CRUISE_Z", "1.5"))
    # A* koridordan çok sapınca episode bitir (0 = kapat)
    PATH_ERROR_TERMINATE_M = float(os.environ.get("ROUTE_PATH_ERROR_TERMINATE", "8.0"))
    PATH_ERROR_TERMINATE_PENALTY = -50.0
    YAW_RESIDUAL_SCALE = 0.25  # dyaw bu kadar çarpanla lookahead yaw'a eklenir

    # Tehdit kapısı: max(/threat/target_scores) → sigmoid; SAC rezidüeli ve path cezası ölçeklenir
    THREAT_GATE_ENABLED = os.environ.get("ROUTE_THREAT_GATE", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    THREAT_GATE_THRESHOLD = float(os.environ.get("ROUTE_THREAT_GATE_THRESHOLD", "0.3"))
    THREAT_GATE_K = float(os.environ.get("ROUTE_THREAT_GATE_K", "5.0"))
    PATH_ERR_SCALE_LOW_THREAT = float(
        os.environ.get("ROUTE_PATH_ERR_SCALE_LOW_THREAT", "0.8")
    )
    PATH_ERR_SCALE_HIGH_THREAT = float(
        os.environ.get("ROUTE_PATH_ERR_SCALE_HIGH_THREAT", "0.1")
    )
    ASTAR_ON_PATH_M = float(os.environ.get("ROUTE_ASTAR_ON_PATH_M", "1.5"))
    ASTAR_RETURN_BONUS = float(os.environ.get("ROUTE_ASTAR_RETURN_BONUS", "0.05"))
    ASTAR_RETURN_GATE_MAX = float(os.environ.get("ROUTE_ASTAR_RETURN_GATE_MAX", "0.2"))
    # Stage 1 (x bölgesi): kapı girdisine [0, MAX] arası uniform gürültü (hedef skorlar düşükken bile sapma eğitimi)
    SYNTHETIC_GATE_STAGE1_MAX = float(
        os.environ.get("ROUTE_SYNTHETIC_GATE_STAGE1_MAX", "0")
    )
    # Hibrit modda dz kullanımı: CRUISE_Z ± clip; 0=sabit CRUISE_Z (eski davranış)
    HYBRID_USE_Z_ACTION = os.environ.get("ROUTE_HYBRID_USE_Z_ACTION", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    HYBRID_Z_CLIP = float(os.environ.get("ROUTE_HYBRID_Z_CLIP", "0.5"))

    ACTOR_COLLISION_RADIUS = 1.5
    ACTOR_COLLISION_Z_MAX = 3.5

    def __init__(self, curriculum_stage=1):
        super().__init__()

        self.curriculum_stage = curriculum_stage
        self.unified_maze = UNIFIED_MAZE
        if self.unified_maze:
            self._stage_origin = UNIFIED_ORIGIN
        else:
            self._stage_origin = STAGE_ORIGINS[curriculum_stage]
        self._spawn_z = 1.5

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "costmap_patch": spaces.Box(
                low=0.0, high=1.0, shape=(1, 64, 64), dtype=np.float32
            ),
            "threat_vector": spaces.Box(
                low=-np.inf, high=np.inf, shape=(74,), dtype=np.float32
            ),
            "threat_scores": spaces.Box(
                low=0.0, high=1.0, shape=(5,), dtype=np.float32
            ),
            "goal_state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            ),
            "a_star_path": spaces.Box(
                low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
            ),
            "privileged": spaces.Box(
                low=-np.inf, high=np.inf, shape=(PRIV_DIM,), dtype=np.float32
            ),
        })

        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.drone_yaw = 0.0
        self.drone_speed = 0.0

        self.goal_x = None
        self.goal_y = None
        self.goal_z = None

        self.prev_dist_to_goal = None
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0

        self.costmap_patch = np.zeros((1, 64, 64), dtype=np.float32)
        self.threat_vector = np.zeros(74, dtype=np.float32)
        self.threat_scores = np.zeros(5, dtype=np.float32)
        self.a_star_poses = []
        self.latest_costmap = None
        self.latest_global_costmap = None

        self.new_patch = False
        self.new_threat = False
        self.new_odom = False
        self._last_odom_time = 0.0
        # reset() sonrası yalnızca bu andan sonra gelen odom "taze" sayılır (teleport sonrası)
        self._reset_wall_time = 0.0
        # Son threat mesajının işlendiği wall time (GIL yükünde edge bayrağı kaçabilir)
        self._last_threat_wall_time = 0.0
        self.cond = threading.Condition()

        self.bridge = CvBridge()

        # Actor tracking (Stage 3)
        self.actor_trajectories = []
        self.actor_ref_time = time.time()

        # Episode stats
        self.episode_reward = 0.0
        self.episode_collisions = 0
        self.episode_successes = 0

        # train_route_curriculum SyncSB3TimestepCallback ile güncellenir (Subproc dahil)
        self._sb3_num_timesteps = 0

        # --- ROS 2 ---
        # Note: rclpy.init() is process-global. Multi-process RL (e.g. SubprocVecEnv)
        # shares the same rclpy context; close() calling rclpy.shutdown() kills others.
        self._rclpy_initialized = False
        if not rclpy.ok():
            rclpy.init()
            self._rclpy_initialized = True

        self.node = rclpy.create_node("route_curriculum_env_node")
        self._running = True

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE, depth=10,
        )
        qos_transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1,
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE, depth=10,
        )
        # Nav2 costmap yayınları: RELIABLE + TRANSIENT_LOCAL (VOLATILE abone eşleşmeyebilir)
        qos_costmap = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10,
        )

        self.node.create_subscription(
            Image, "/route/costmap_patch", self._cb_patch, 10)
        self.node.create_subscription(
            Float32MultiArray, "/threat/state_vec", self._cb_threat, 10)
        self.node.create_subscription(
            Float32MultiArray, "/threat/target_scores", self._cb_threat_scores, 10)
        self.node.create_subscription(
            Odometry, "/odometry/filtered", self._cb_odom, qos_sensor)
        self.node.create_subscription(
            PoseStamped, "/goal_pose", self._cb_goal, qos_transient)
        self.node.create_subscription(
            OccupancyGrid, "/local_costmap/costmap", self._cb_costmap, qos_costmap)
        self.node.create_subscription(
            OccupancyGrid, "/global_costmap/costmap", self._cb_global_costmap, qos_costmap)
        self.node.create_subscription(
            Path, "/plan", self._cb_plan, qos_sensor)

        self.waypoint_pub = self.node.create_publisher(
            PoseStamped, "/route/waypoint_desired", 10)

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        time.sleep(1.0)  # Allow spin thread to subscribe and receive first messages

        if self.curriculum_stage == 3 or self.unified_maze:
            self._load_actor_data()

    # ------------------------------------------------------------------ #
    # Curriculum API
    # ------------------------------------------------------------------ #
    def _stage_from_position(self, x: float) -> int:
        """Pozisyona gore stage (birlesik maze)."""
        if x < SECTION1_X_MAX:
            return 1
        if x < SECTION2_X_MAX:
            return 2
        return 3

    def set_curriculum_stage(self, stage: int):
        self.curriculum_stage = stage
        if self.unified_maze:
            self._stage_origin = UNIFIED_ORIGIN
        else:
            self._stage_origin = STAGE_ORIGINS[stage]
        self.node.get_logger().info(
            f"Curriculum stage → {stage}  origin={self._stage_origin}"
        )
        if stage == 3 or self.unified_maze:
            self._load_actor_data()
        else:
            self.actor_trajectories = []

    # ------------------------------------------------------------------ #
    # ROS spin & callbacks
    # ------------------------------------------------------------------ #
    def _spin(self):
        while self._running and rclpy.ok():
            try:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            except Exception:
                break

    def _cb_patch(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            cv_img = cv2.resize(cv_img, (64, 64))
            normalized = cv_img.astype(np.float32) / 255.0
            with self.cond:
                self.costmap_patch[0] = normalized
                self.new_patch = True
                self.cond.notify_all()
        except Exception:
            pass

    def _cb_threat(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        if data.shape[0] == 74:
            with self.cond:
                self.threat_vector = data
                self.new_threat = True
                self._last_threat_wall_time = time.time()
                self.cond.notify_all()

    def _cb_threat_scores(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        n = min(len(data), 5)
        with self.cond:
            self.threat_scores[:n] = data[:n]
            if n < 5:
                self.threat_scores[n:] = 0.0
            self.cond.notify_all()

    def _cb_odom(self, msg: Odometry):
        with self.cond:
            self.drone_x = msg.pose.pose.position.x
            self.drone_y = msg.pose.pose.position.y
            self.drone_z = msg.pose.pose.position.z

            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            self.drone_speed = math.sqrt(vx * vx + vy * vy)

            q = msg.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.drone_yaw = math.atan2(siny, cosy)

            self.new_odom = True
            self._last_odom_time = time.time()
            self.cond.notify_all()

    def _cb_goal(self, msg: PoseStamped):
        with self.cond:
            self.goal_x = msg.pose.position.x
            self.goal_y = msg.pose.position.y
            self.goal_z = msg.pose.position.z

    def _cb_costmap(self, msg: OccupancyGrid):
        with self.cond:
            self.latest_costmap = msg

    def _cb_global_costmap(self, msg: OccupancyGrid):
        with self.cond:
            self.latest_global_costmap = msg

    def _cb_plan(self, msg: Path):
        poses = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self.cond:
            self.a_star_poses = poses

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _wait_obs(self, timeout: float = 0.5, step_mode: bool = False) -> bool:
        """step_mode=True: SAC+PyTorch ana thread GIL yüzünden spin gecikince bile
        son threat verisi yeterli sayılır (yeni edge + patch + odom şartı çok sıkı kalırdı)."""
        end = time.time() + timeout
        threat_grace_sec = 0.5
        with self.cond:
            while time.time() < end:
                odom_age = time.time() - self._last_odom_time
                odom_after_reset = self._last_odom_time >= self._reset_wall_time
                odom_fresh = odom_age < 0.35 and odom_after_reset
                threat_ok = self.new_threat
                if step_mode and not threat_ok:
                    threat_ok = (time.time() - self._last_threat_wall_time) < threat_grace_sec
                if self.new_patch and threat_ok and odom_fresh:
                    self.new_patch = False
                    self.new_threat = False
                    return True
                remaining = end - time.time()
                if remaining > 0:
                    self.cond.wait(timeout=remaining)
        missing = []
        if not self.new_patch:
            missing.append("/route/costmap_patch")
        threat_recent = (time.time() - self._last_threat_wall_time) < threat_grace_sec
        if not self.new_threat and not (step_mode and threat_recent):
            missing.append("/threat/state_vec")
        odom_age = time.time() - self._last_odom_time
        if self._last_odom_time <= 0.0:
            missing.append("/odometry/filtered (hiç mesaj yok veya sim düştü)")
        elif self._last_odom_time < self._reset_wall_time:
            missing.append("/odometry/filtered (reset/teleport sonrası yeni mesaj gelmedi)")
        elif odom_age >= 0.35:
            missing.append(f"/odometry/filtered (son: {odom_age:.2f}s önce — kopuk olabilir)")
        self.node.get_logger().warn(
            f"Obs timeout ({timeout:.1f}s): eksik: {', '.join(missing)}"
        )
        return False

    def _build_goal_state(self) -> np.ndarray:
        with self.cond:
            gx, gy, gz = self.goal_x, self.goal_y, self.goal_z
            dx, dy, dz = self.drone_x, self.drone_y, self.drone_z
            dyaw = self.drone_yaw
            dspeed = self.drone_speed
        if gx is None:
            return np.zeros(7, dtype=np.float32)

        rel_x = gx - dx
        rel_y = gy - dy
        rel_z = (gz if gz is not None else dz) - dz
        dist = math.sqrt(rel_x * rel_x + rel_y * rel_y)
        dist_norm = min(dist / self.MAX_GOAL_DIST, 1.0)
        speed_norm = min(dspeed / self.MAX_SPEED, 1.0)

        return np.array([
            rel_x, rel_y, rel_z,
            dist_norm, speed_norm,
            math.sin(dyaw),
            math.cos(dyaw),
        ], dtype=np.float32)

    def get_drone_position(self):
        """Thread-safe drone position for callbacks (e.g. TrajectoryRecorder)."""
        with self.cond:
            return (self.drone_x, self.drone_y, self.drone_z)

    @staticmethod
    def _astar_forward_start_index(poses, dx: float, dy: float) -> int:
        """İlk 'ileri' waypoint indeksi: geçilmiş segmentleri atlar (_build_a_star_obs ile aynı)."""
        if not poses:
            return 0
        nearest_idx = 0
        best_dist_sq = float("inf")
        for i, (px, py) in enumerate(poses):
            d_sq = (px - dx) ** 2 + (py - dy) ** 2
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                nearest_idx = i

        # Forward-looking: skip waypoints already passed (dot product with path dir)
        if nearest_idx + 1 < len(poses):
            p0, p1 = poses[nearest_idx], poses[nearest_idx + 1]
            path_dx = p1[0] - p0[0]
            path_dy = p1[1] - p0[1]
            norm = math.sqrt(path_dx ** 2 + path_dy ** 2) + 1e-9
            path_dx /= norm
            path_dy /= norm
            drone_to_wp_x = p0[0] - dx
            drone_to_wp_y = p0[1] - dy
            dot = drone_to_wp_x * path_dx + drone_to_wp_y * path_dy
            return nearest_idx + 1 if dot < 0 else nearest_idx
        return nearest_idx

    @staticmethod
    def _threat_sigmoid_gate(max_score: float, threshold: float, k: float) -> float:
        """max(target_scores) ∈ [0,1] → tehdit kapısı ∈ (0,1). k büyüdükçe geçiş keskinleşir."""
        t = float(np.clip(max_score, 0.0, 1.0))
        if k <= 0.0:
            return 1.0 if t >= threshold else 0.0
        x = k * (t - threshold)
        if x > 40.0:
            return 1.0
        if x < -40.0:
            return 0.0
        return float(1.0 / (1.0 + math.exp(-x)))

    def _build_a_star_obs(self) -> np.ndarray:
        result = np.zeros(self.NUM_PATH_WPS * 2, dtype=np.float32)
        with self.cond:
            poses = list(self.a_star_poses)
            dx, dy = self.drone_x, self.drone_y
        if not poses:
            return result

        start = self._astar_forward_start_index(poses, dx, dy)

        count = 0
        for i in range(start, len(poses)):
            if count >= self.NUM_PATH_WPS:
                break
            wx, wy = poses[i]
            result[count * 2] = wx - dx
            result[count * 2 + 1] = wy - dy
            count += 1

        return result

    def _build_global_costmap_patch(self) -> np.ndarray:
        """64x64 patch from global costmap centered on drone (map frame).
        Returns (1, 64, 64) normalized [0,1]. Unknown=0.5, free=0, occupied=1.
        Used for privileged obs (critic only). Call from _build_privileged when ready.
        """
        patch = np.full((1, 64, 64), 0.5, dtype=np.float32)  # unknown default
        with self.cond:
            cm = self.latest_global_costmap
            drone_x, drone_y = self.drone_x, self.drone_y
        if cm is None:
            return patch

        res = cm.info.resolution
        ox = cm.info.origin.position.x
        oy = cm.info.origin.position.y
        w = cm.info.width
        h = cm.info.height

        # Drone position in map frame (odom≈map in typical sim)
        drone_px = int((drone_x - ox) / res)
        drone_py = int((drone_y - oy) / res)

        half = 32
        x_start = drone_px - half
        x_end = drone_px + half
        y_start = drone_py - half
        y_end = drone_py + half

        data = np.array(cm.data, dtype=np.int8).reshape((h, w))
        patch_uint8 = np.full((64, 64), 128, dtype=np.uint8)

        src_x_s = max(0, x_start)
        src_x_e = min(w, x_end)
        src_y_s = max(0, y_start)
        src_y_e = min(h, y_end)
        dst_x_s = src_x_s - x_start
        dst_x_e = dst_x_s + (src_x_e - src_x_s)
        dst_y_s = src_y_s - y_start
        dst_y_e = dst_y_s + (src_y_e - src_y_s)

        if src_x_e > src_x_s and src_y_e > src_y_s:
            raw = data[src_y_s:src_y_e, src_x_s:src_x_e]
            out = np.full(raw.shape, 128, dtype=np.uint8)
            out[raw == -1] = 128
            out[raw >= 0] = (np.clip(raw[raw >= 0], 0, 100) * 2.55).astype(np.uint8)
            patch_uint8[dst_y_s:dst_y_e, dst_x_s:dst_x_e] = out

        patch[0] = patch_uint8.astype(np.float32) / 255.0
        return patch

    def _build_privileged(self) -> np.ndarray:
        """Privileged obs for critic (training only). Actor ignores at inference."""
        priv = np.zeros(PRIV_DIM, dtype=np.float32)
        idx = 0

        # Snapshot all cond-protected state once for consistency (actor rel, path rel)
        with self.cond:
            dx, dy = self.drone_x, self.drone_y
            lidar = self.threat_vector[self.LIDAR_START_IDX : self.LIDAR_END_IDX].copy()
            poses = list(self.a_star_poses)
            prev_action = self.prev_action.copy()

        # [0:64] Global costmap embedding (8x8 downsampled)
        global_patch = self._build_global_costmap_patch()[0]
        for i in range(0, 64, 8):
            for j in range(0, 64, 8):
                priv[idx] = float(np.mean(global_patch[i : i + 8, j : j + 8]))
                idx += 1
        assert idx == 64

        # [64:74] Actor positions rel (5 x,y) — drone frame
        actor_positions = self._compute_actor_positions()
        for ax, ay in actor_positions[:5]:
            priv[idx] = ax - dx
            priv[idx + 1] = ay - dy
            idx += 2
        assert idx <= 74, f"actor slot overflow: idx={idx}"
        idx = 74

        # [74:78] Collision risks (lidar_min, actor_min, collision_binary, reserved)
        lidar_min = float(np.min(lidar)) * self.LIDAR_MAX_RANGE if len(lidar) > 0 else 30.0
        actor_min = (
            min(math.sqrt((ax - dx) ** 2 + (ay - dy) ** 2) for ax, ay in actor_positions)
            if actor_positions else 30.0
        )
        priv[74] = np.clip(lidar_min / 10.0, 0.0, 1.0)
        priv[75] = np.clip(actor_min / 10.0, 0.0, 1.0)
        # Current-position collision check (semantic: "am I colliding right now?")
        priv[76] = 1.0 if self._check_collision(dx, dy) else 0.0
        priv[77] = 0.0
        idx = 78

        # [78:98] Full path ahead (10 wp x,y)
        for i in range(min(10, len(poses))):
            wx, wy = poses[i]
            priv[idx] = wx - dx
            priv[idx + 1] = wy - dy
            idx += 2
        idx = 98

        # [98:102] Prev action
        priv[98:102] = prev_action
        return priv

    def _get_obs(self) -> dict:
        with self.cond:
            patch = self.costmap_patch.copy()
            threat_vec = self.threat_vector.copy()
            threat_sc = self.threat_scores.copy()
        return {
            "costmap_patch": patch,
            "threat_vector": threat_vec,
            "threat_scores": np.clip(threat_sc, 0.0, 1.0),
            "goal_state": self._build_goal_state(),
            "a_star_path": self._build_a_star_obs(),
            "privileged": self._build_privileged(),
        }

    def _dist_to_goal(self) -> float:
        with self.cond:
            gx, gy = self.goal_x, self.goal_y
            dx, dy = self.drone_x, self.drone_y
        if gx is None:
            return float("inf")
        return math.sqrt((gx - dx) ** 2 + (gy - dy) ** 2)

    def _dist_to_nearest_astar_wp(self) -> float:
        with self.cond:
            poses = list(self.a_star_poses)
            dx, dy = self.drone_x, self.drone_y
        if not poses:
            return 0.0
        best_sq = float("inf")
        for px, py in poses:
            d_sq = (px - dx) ** 2 + (py - dy) ** 2
            if d_sq < best_sq:
                best_sq = d_sq
        return math.sqrt(best_sq)

    # ------------------------------------------------------------------ #
    # Collision detection
    # ------------------------------------------------------------------ #
    def _check_collision(self, wp_x: float, wp_y: float) -> bool:
        # Costmap: hedef hücre + drone anlık konumu (yolda duvara girme)
        with self.cond:
            dx, dy = self.drone_x, self.drone_y
        if self._check_costmap_collision(wp_x, wp_y) or self._check_costmap_collision(
            dx, dy
        ):
            return True
        if self._check_lidar_collision():
            return True
        stage = self._stage_from_position(self.drone_x) if self.unified_maze else self.curriculum_stage
        if stage == 3 and self._check_actor_collision():
            return True
        return False

    def _check_costmap_collision(self, x: float, y: float) -> bool:
        with self.cond:
            cm = self.latest_costmap
        if cm is None:
            return False
        res = cm.info.resolution
        ox = cm.info.origin.position.x
        oy = cm.info.origin.position.y
        w = cm.info.width
        h = cm.info.height

        px = int((x - ox) / res)
        py = int((y - oy) / res)
        if not (0 <= px < w and 0 <= py < h):
            return False

        idx = py * w + px
        return cm.data[idx] >= self.LETHAL_THRESHOLD

    def _check_lidar_collision(self) -> bool:
        lidar = self.threat_vector[self.LIDAR_START_IDX:self.LIDAR_END_IDX]
        if len(lidar) == 0:
            return False
        min_dist_m = float(np.min(lidar)) * self.LIDAR_MAX_RANGE
        return min_dist_m < self.LIDAR_COLLISION_M

    def _check_actor_collision(self) -> bool:
        if not self.actor_trajectories:
            return False
        if self.drone_z > self.ACTOR_COLLISION_Z_MAX:
            return False

        for ax, ay in self._compute_actor_positions():
            dx = self.drone_x - ax
            dy = self.drone_y - ay
            if math.sqrt(dx * dx + dy * dy) < self.ACTOR_COLLISION_RADIUS:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Actor trajectory (analytical)
    # ------------------------------------------------------------------ #
    def _load_actor_data(self):
        path = ACTORS_UNIFIED_JSON_PATH if self.unified_maze else ACTORS_JSON_PATH
        try:
            with open(path) as f:
                data = json.load(f)
            self.actor_trajectories = data.get("actors", [])
            self.actor_ref_time = data.get("spawn_time", time.time())
            self.node.get_logger().info(
                f"Actor data loaded: {len(self.actor_trajectories)} actors"
            )
        except FileNotFoundError:
            self.actor_trajectories = []
        except Exception as e:
            self.node.get_logger().warn(f"Actor data load error: {e}")
            self.actor_trajectories = []

    def _compute_actor_positions(self):
        """Actor positions from JSON trajectory. Period must be >= 2*duration+0.5
        for full round-trip; else actor may not return to (x1,y1)."""
        elapsed = time.time() - self.actor_ref_time
        positions = []
        for a in self.actor_trajectories:
            x1, y1 = a["x1"], a["y1"]
            x2, y2 = a["x2"], a["y2"]
            period = max(a.get("period", 10.0), 1e-6)
            speed = max(a.get("speed", 1.0), 1e-6)
            dx, dy = x2 - x1, y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                positions.append((x1, y1))
                continue
            duration = dist / speed
            min_period = 2.0 * duration + 0.5
            if period < min_period:
                period = min_period
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

    # ------------------------------------------------------------------ #
    # Teleportation / soft reset
    # ------------------------------------------------------------------ #
    def _soft_reset_drone(self):
        if self.unified_maze:
            x, y = UNIFIED_ORIGIN
        else:
            x, y = self._stage_origin
        z = self._spawn_z
        req_str = (
            f'name: "{self.DRONE_MODEL_NAME}", '
            f"position: {{x: {x}, y: {y}, z: {z}}}, "
            "orientation: {w: 1.0}"
        )
        try:
            subprocess.run(
                [
                    "gz", "service", "-s", f"/world/{self.GZ_WORLD_NAME}/set_pose",
                    "--reqtype", "gz.msgs.Pose",
                    "--reptype", "gz.msgs.Boolean",
                    "--timeout", "1000",
                    "--req", req_str,
                ],
                shell=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except Exception as e:
            self.node.get_logger().warn(f"Soft reset error: {e}")

        time.sleep(self.RESET_STABILIZE_SEC)

    def _publish_waypoint(self, wx: float, wy: float, wz: float, wyaw: float):
        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.pose.position.x = wx
        msg.pose.position.y = wy
        msg.pose.position.z = wz
        msg.pose.orientation.z = math.sin(wyaw / 2.0)
        msg.pose.orientation.w = math.cos(wyaw / 2.0)
        self.waypoint_pub.publish(msg)

    # ------------------------------------------------------------------ #
    # Gym interface
    # ------------------------------------------------------------------ #
    def set_sb3_timesteps(self, n: int) -> None:
        """Stable-Baselines3 toplam environment adımı (learn döngüsü). Callback yazar."""
        try:
            self._sb3_num_timesteps = max(0, int(n))
        except (TypeError, ValueError):
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_collisions = 0
        self.episode_successes = 0

        with self.cond:
            self.new_patch = False
            self.new_threat = False
            self.new_odom = False

        self._soft_reset_drone()
        # Teleport + stabilize bittikten sonra: bundan sonra gelen odom gerekli (_last_odom_time sıfırlanmaz)
        with self.cond:
            self._reset_wall_time = time.time()

        obs_ok = False
        for attempt in range(5):
            obs_ok = self._wait_obs(timeout=2.0)
            if obs_ok:
                break
            self.node.get_logger().warn(f"Reset obs retry {attempt + 1}/5")
            time.sleep(0.5)

        if not obs_ok:
            self.node.get_logger().error("Reset sonrası obs alınamadı, sıfır obs ile devam.")

        self.prev_dist_to_goal = self._dist_to_goal()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)
        # SAC: learning_starts öncesi rastgele aksiyon → waypoint sarsıntısı. İsteğe bağlı ölçekle.
        _ls = int(os.environ.get("ROUTE_LEARNING_STARTS", "5000"))
        _rscale = float(os.environ.get("ROUTE_RANDOM_PHASE_ACTION_SCALE", "1.0"))
        if _rscale < 1.0 - 1e-9 and self._sb3_num_timesteps < _ls:
            action = np.clip(action.astype(np.float32) * _rscale, -1.0, 1.0)

        with self.cond:
            max_threat_pre = float(np.max(self.threat_scores))
            px = self.drone_x

        stage_for_gate = (
            self._stage_from_position(px) if self.unified_maze else self.curriculum_stage
        )
        synth_delta = 0.0
        effective_threat = max_threat_pre
        if self.SYNTHETIC_GATE_STAGE1_MAX > 0.0 and stage_for_gate == 1:
            synth_delta = float(
                self.np_random.uniform(0.0, self.SYNTHETIC_GATE_STAGE1_MAX)
            )
            effective_threat = min(1.0, max_threat_pre + synth_delta)

        threat_gate = self._threat_sigmoid_gate(
            effective_threat,
            self.THREAT_GATE_THRESHOLD,
            self.THREAT_GATE_K,
        )
        gate_apply = threat_gate if self.THREAT_GATE_ENABLED else 1.0

        dx = float(action[0]) * self.STEP_SIZE
        dy = float(action[1]) * self.STEP_SIZE
        dz = float(action[2]) * self.Z_STEP
        dyaw = float(action[3]) * self.MAX_YAW_RATE

        cos_yaw = math.cos(self.drone_yaw)
        sin_yaw = math.sin(self.drone_yaw)
        world_dx = cos_yaw * dx - sin_yaw * dy
        world_dy = sin_yaw * dx + cos_yaw * dy
        world_dx *= gate_apply
        world_dy *= gate_apply

        with self.cond:
            poses = list(self.a_star_poses)
            ddx, ddy = self.drone_x, self.drone_y

        use_hybrid = self.HYBRID_ASTAR_BASELINE and bool(poses)
        if use_hybrid:
            start = self._astar_forward_start_index(poses, ddx, ddy)
            lx, ly = poses[start]
            wp_x = lx + world_dx
            wp_y = ly + world_dy
            if self.HYBRID_USE_Z_ACTION:
                dz_scaled = float(action[2]) * self.Z_STEP * gate_apply
                wp_z = float(
                    np.clip(
                        self.CRUISE_Z + dz_scaled,
                        self.CRUISE_Z - self.HYBRID_Z_CLIP,
                        self.CRUISE_Z + self.HYBRID_Z_CLIP,
                    )
                )
            else:
                wp_z = self.CRUISE_Z
            rel_lx = lx - ddx
            rel_ly = ly - ddy
            if rel_lx * rel_lx + rel_ly * rel_ly > 0.01:
                wp_yaw = math.atan2(rel_ly, rel_lx) + float(
                    action[3]
                ) * self.MAX_YAW_RATE * self.YAW_RESIDUAL_SCALE * gate_apply
            else:
                wp_yaw = self.drone_yaw + dyaw * gate_apply
        else:
            wp_x = ddx + world_dx
            wp_y = ddy + world_dy
            wp_z = self.drone_z + dz * gate_apply
            wp_yaw = self.drone_yaw + dyaw * gate_apply

        self._publish_waypoint(wp_x, wp_y, wp_z, wp_yaw)
        # GIL: eğitim sırasında spin thread gecikir; threat edge kaçabilir → step_mode + biraz daha uzun süre.
        obs_sync_ok = self._wait_obs(timeout=0.40, step_mode=True)

        # Birlesik maze: stage pozisyondan hesapla
        if self.unified_maze:
            self.curriculum_stage = self._stage_from_position(self.drone_x)

        # ══════════════════════════════════════════════════════════════
        # FIXED REWARD (identical formula across all stages)
        # ══════════════════════════════════════════════════════════════
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "stage": self.curriculum_stage,
            "obs_sync_ok": obs_sync_ok,
            "threat_gate": threat_gate,
            "threat_gate_applied": gate_apply,
            "threat_effective_max": effective_threat,
            "synthetic_threat_delta": synth_delta,
            "stage_for_gate": stage_for_gate,
        }

        dist = self._dist_to_goal()

        rw_progress = 0.0
        rw_goal = 0.0
        rw_collision = 0.0
        rw_path = 0.0
        rw_astar_return = 0.0
        rw_path_drift = 0.0
        rw_threat = 0.0
        rw_smooth = 0.0
        rw_time = -0.1

        # r_progress
        if self.prev_dist_to_goal is not None and self.goal_x is not None:
            progress = self.prev_dist_to_goal - dist
            rw_progress = 3.0 * progress
            reward += rw_progress

        # r_goal vs r_collision: collision takes precedence (elif)
        if self._check_collision(wp_x, wp_y):
            rw_collision = -100.0
            reward += rw_collision
            terminated = True
            info["collision"] = True
            self.episode_collisions += 1
        elif dist < self.GOAL_TOLERANCE:
            rw_goal = 50.0
            reward += rw_goal
            terminated = True
            info["success"] = True
            self.episode_successes += 1

        # r_path_error: tehdit kapısı açıkken path_w ile ölçeklenir
        path_err = self._dist_to_nearest_astar_wp()
        path_err_normalized = min(path_err / 5.0, 1.0)
        on_astar = 1.0 if path_err < self.ASTAR_ON_PATH_M else 0.0
        info["on_astar"] = on_astar
        if self.THREAT_GATE_ENABLED:
            path_w = (
                self.PATH_ERR_SCALE_LOW_THREAT * (1.0 - threat_gate)
                + self.PATH_ERR_SCALE_HIGH_THREAT * threat_gate
            )
            rw_path = -0.5 * path_err_normalized * path_w
            reward += rw_path
            if (
                threat_gate < self.ASTAR_RETURN_GATE_MAX
                and path_err < self.ASTAR_ON_PATH_M
            ):
                rw_astar_return = self.ASTAR_RETURN_BONUS
                reward += rw_astar_return
        else:
            rw_path = -0.5 * path_err_normalized
            reward += rw_path

        # Çok büyük A* sapması → episode bitir (ROUTE_PATH_ERROR_TERMINATE=0 ile kapat)
        if not terminated and self.PATH_ERROR_TERMINATE_M > 0.0:
            with self.cond:
                has_plan = len(self.a_star_poses) >= 2
            if has_plan and path_err > self.PATH_ERROR_TERMINATE_M:
                rw_path_drift = self.PATH_ERROR_TERMINATE_PENALTY
                reward += rw_path_drift
                terminated = True
                info["path_drift_terminate"] = True

        # r_threat_proximity (adım sonrası skor; kapı hesabı yukarıda max_threat_pre ile)
        max_threat = float(np.max(self.threat_scores))
        rw_threat = -2.0 * max_threat
        reward += rw_threat

        # r_smooth
        action_delta = float(np.linalg.norm(action - self.prev_action))
        rw_smooth = -0.05 * action_delta
        reward += rw_smooth

        # r_time
        reward += rw_time

        # truncation
        if self.step_count >= self.MAX_EPISODE_STEPS:
            truncated = True
            info["timeout"] = True

        self.prev_dist_to_goal = dist
        self.prev_action = action.copy()
        self.episode_reward += reward

        info["path_error"] = path_err
        info["max_threat"] = max_threat
        info["dist_to_goal"] = dist
        info["episode_reward"] = self.episode_reward
        info["episode_collisions"] = self.episode_collisions
        info["episode_successes"] = self.episode_successes
        info["rw_progress"] = rw_progress
        info["rw_goal"] = rw_goal
        info["rw_collision"] = rw_collision
        info["rw_path"] = rw_path
        info["rw_astar_return"] = rw_astar_return
        info["rw_path_drift"] = rw_path_drift
        info["rw_threat"] = rw_threat
        info["rw_smooth"] = rw_smooth
        info["rw_time"] = rw_time

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self._running = False
        time.sleep(0.2)
        if self.node:
            self.node.destroy_node()
        if getattr(self, "_rclpy_initialized", False) and rclpy.ok():
            rclpy.shutdown()
            