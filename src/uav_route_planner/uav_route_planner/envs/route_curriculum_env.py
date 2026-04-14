#!/usr/bin/env python3
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

from std_msgs.msg import Float32MultiArray, String, Float32
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseArray

_WALLS_DIR = "/home/ubuntu/Desktop"
STAGE_ORIGINS = {1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, 0.0)}
WALLS_PATHS = {
    1: os.path.join(_WALLS_DIR, "maze_walls.json"),
    2: os.path.join(_WALLS_DIR, "maze_walls.json"),
    3: os.path.join(_WALLS_DIR, "maze_walls.json"),
}
ACTORS_JSON_PATH = os.path.join(_WALLS_DIR, "maze_actors_stage3.json")
ACTORS_UNIFIED_JSON_PATH = os.path.join(_WALLS_DIR, "maze_actors_unified.json")
WALLS_UNIFIED_PATH = os.path.join(_WALLS_DIR, "maze_walls_unified.json")

UNIFIED_MAZE = False 
UNIFIED_ORIGIN = (0.0, 0.0)
SECTION1_X_MAX = 5000.0
SECTION2_X_MAX = 12500.0

class RouteCurriculumEnv(gym.Env):
    metadata = {"render_modes": []}
    STEP_SIZE = float(os.environ.get("ROUTE_STEP_SIZE", "0.3"))
    Z_STEP = 0.2
    MAX_YAW_RATE = 0.52
    GOAL_TOLERANCE = 1.3
    SAFETY_RADIUS = 0.5
    LETHAL_THRESHOLD = 90
    MAX_EPISODE_STEPS = 2048
    MAX_GOAL_DIST = 30.0
    MAX_SPEED = 5.0

    LIDAR_MAX_RANGE = 30.0
    LIDAR_COLLISION_M = 0.85
    LIDAR_START_IDX = 3
    LIDAR_END_IDX = 39
    # Soft wall-avoidance shaping using 36-sector lidar slice [3:39].
    # Applies before hard collision threshold so RL learns to keep margin.
    WALL_SOFT_DIST_M = float(os.environ.get("ROUTE_WALL_SOFT_DIST_M", "1.5"))
    WALL_PROX_PENALTY_SCALE = float(os.environ.get("ROUTE_WALL_PROX_PENALTY_SCALE", "1.0"))
    NUM_PATH_WPS = 5

    # ── Threat token layout (threat_vector[39:74] = 5 × 7) ──────────────────
    THREAT_TOKEN_OFFSET  = 39   # first token starts here in threat_vector
    THREAT_TOKEN_SIZE    = 7
    THREAT_NUM_OBJECTS   = 5
    # Field indices within each 7-element token
    TK_CLASS_ID   = 0
    TK_DIST_NORM  = 1   # normalized dist; multiply by LIDAR_MAX_RANGE for metres
    TK_CLOSING    = 2   # closing speed (positive = approaching)
    TK_SIN_BEAR   = 3   # sin(bearing) in drone frame
    TK_COS_BEAR   = 4   # cos(bearing) in drone frame
    TK_CONFIDENCE = 5
    TK_VALID      = 6   # 1.0 = valid object

    # ── Evasion reward gating & scales ──────────────────────────────────────
    EVASION_THREAT_GATE    = 0.4    # min max_threat score to activate evasion rewards
    EVASION_DIST_MAX_M     = 8.0    # only reward evasion when threat closer than this
    THREAT_AWAY_SCALE      = 1.0    # reward per metre of dist increase from nearest threat
    LATERAL_ESCAPE_SCALE   = 0.5    # reward per metre of lateral movement vs threat axis
    LATERAL_MIN_MOVE_M     = 0.03   # min drone movement to compute lateral escape

    # ── Dormant RL state machine thresholds ──────────────────────────────────
    RL_D_ACTIVATE    = 6.0    # m  — activate when nearest threat closer than this
    RL_D_RELEASE     = 10.0   # m  — deactivate (hysteresis gap, must be > RL_D_ACTIVATE)
    RL_RISK_ACTIVATE = 0.45   # composite risk score to wake RL
    RL_RISK_RELEASE  = 0.20   # composite risk score to sleep RL
    RL_RISK_W_DIST   = 0.5    # weight for distance component in risk score
    RL_RISK_W_THREAT = 0.5    # weight for threat score component in risk score
    # Fallback wake-up for cases where threat token distance is missing/stale but
    # target scores clearly indicate danger (prevents permanent dormant mode).
    RL_SCORE_ACTIVATE = float(os.environ.get("ROUTE_RL_SCORE_ACTIVATE", "0.23"))

    DRONE_MODEL_NAME = "x500_mono_cam_0"
    GZ_WORLD_NAME = "default"
    RESET_STABILIZE_SEC = 2.0 
    HYBRID_ASTAR_BASELINE = os.environ.get("ROUTE_HYBRID_ASTAR", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    CRUISE_Z = float(os.environ.get("ROUTE_CRUISE_Z", "1.5"))
    PATH_ERROR_TERMINATE_M = float(os.environ.get("ROUTE_PATH_ERROR_TERMINATE", "8.0"))
    PATH_ERROR_TERMINATE_PENALTY = -50.0
    YAW_RESIDUAL_SCALE = 0.25  
    THREAT_GATE_ENABLED = os.environ.get("ROUTE_THREAT_GATE", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    THREAT_GATE_THRESHOLD = float(os.environ.get("ROUTE_THREAT_GATE_THRESHOLD", "0.4"))
    THREAT_GATE_K = float(os.environ.get("ROUTE_THREAT_GATE_K", "5.0"))
    PATH_ERR_SCALE_LOW_THREAT = float(
        os.environ.get("ROUTE_PATH_ERR_SCALE_LOW_THREAT", "0.8")
    )
    PATH_ERR_SCALE_HIGH_THREAT = float(
        os.environ.get("ROUTE_PATH_ERR_SCALE_HIGH_THREAT", "0.1")
    )
    ASTAR_ON_PATH_M = float(os.environ.get("ROUTE_ASTAR_ON_PATH_M", "2.5"))
    ASTAR_RETURN_BONUS = float(os.environ.get("ROUTE_ASTAR_RETURN_BONUS", "0.05"))
    ASTAR_RETURN_GATE_MAX = float(os.environ.get("ROUTE_ASTAR_RETURN_GATE_MAX", "0.2"))
    SYNTHETIC_GATE_STAGE1_MAX = float(
        os.environ.get("ROUTE_SYNTHETIC_GATE_STAGE1_MAX", "0")
    )
    HYBRID_USE_Z_ACTION = os.environ.get("ROUTE_HYBRID_USE_Z_ACTION", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    HYBRID_Z_CLIP = float(os.environ.get("ROUTE_HYBRID_Z_CLIP", "0.5"))
    EP_GOAL_MAX_M = float(os.environ.get("ROUTE_EP_GOAL_MAX_M", "5.0"))

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
        self._last_plan_time = 0.0
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "lidar_vector": spaces.Box(
                low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32
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
        })

        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.drone_yaw = 0.0
        self.drone_speed = 0.0

        self.goal_x = None
        self.goal_y = None
        self.goal_z = None
        self._ep_goal_x = None
        self._ep_goal_y = None

        self.prev_dist_to_goal = None
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.smoothed_action = np.zeros(4, dtype=np.float32)
        self.action_ema_alpha = float(os.environ.get("ROUTE_ACTION_EMA_ALPHA", "0.2"))
        self.step_count = 0

        self.threat_vector = np.zeros(74, dtype=np.float32)
        self.threat_scores = np.zeros(5, dtype=np.float32)
        self.a_star_poses = []
        self.latest_costmap = None

        self.new_threat = False
        self.new_odom = False
        self._last_odom_time = 0.0
        self._reset_wall_time = 0.0
        self._last_threat_wall_time = 0.0
        self.cond = threading.Condition()

        self.latest_actor_poses = []

        self.episode_reward = 0.0
        self.episode_collisions = 0
        self.episode_successes = 0
                # ── ACTIVE-mode episode diagnostics ─────────────────────────
        self._active_step_count = 0
        self._active_progress_sum = 0.0
        self._active_start_goal_dist = None
        self._active_end_goal_dist = None
        self._active_threat_away_sum = 0.0
        self._active_lateral_escape_sum = 0.0

        self._sb3_num_timesteps = 0

        # ── Evasion reward state ─────────────────────────────────────────
        self._prev_drone_x: float = 0.0
        self._prev_drone_y: float = 0.0
        self._prev_nearest_threat_dist: float | None = None  # metres

        # ── Dormant RL state ───────────────────────────────────────────
        self._rl_mode: str = "DORMANT"  # "DORMANT" | "ACTIVE"

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
            durability=DurabilityPolicy.VOLATILE, depth=1,
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE, depth=10,
        )
        qos_costmap = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10,
        )

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
            Path, "/plan", self._cb_plan, qos_sensor)
        self.node.create_subscription(
            PoseArray, "/route/actor_poses", self._cb_actor_poses, 10)

        self.waypoint_pub = self.node.create_publisher(
            PoseStamped, "/route/waypoint_desired", 10)
        # Debug publishers for dormant RL monitoring
        self._rl_mode_pub = self.node.create_publisher(String, "/rl_avoidance/mode", 10)
        self._rl_risk_pub = self.node.create_publisher(Float32, "/rl_avoidance/risk", 10)

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        time.sleep(1.0)  

    def _stage_from_position(self, x: float) -> int:
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

    def _spin(self):
        from rclpy.executors import SingleThreadedExecutor
        executor = SingleThreadedExecutor()
        executor.add_node(self.node)
        try:
            while self._running and rclpy.ok():
                try:
                    executor.spin_once(timeout_sec=0.1)
                except Exception:
                    break
        finally:
            executor.remove_node(self.node)

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
            new_gx = msg.pose.position.x
            new_gy = msg.pose.position.y
            new_gz = msg.pose.position.z
            self.goal_x = new_gx
            self.goal_y = new_gy
            self.goal_z = new_gz

    def _cb_costmap(self, msg: OccupancyGrid):
        with self.cond:
            self.latest_costmap = msg

    def _cb_plan(self, msg: Path):
        poses = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self._last_plan_time = time.time()
        with self.cond:
            self.a_star_poses = poses

    def _cb_actor_poses(self, msg: PoseArray):
        with self.cond:
            self.latest_actor_poses = [(p.position.x, p.position.y) for p in msg.poses]

    def _wait_obs(self, timeout: float = 0.5, step_mode: bool = False) -> bool:
        end = time.time() + timeout
        threat_grace_sec = 0.5
        with self.cond:
            while time.time() < end:
                odom_age = time.time() - self._last_odom_time
                odom_after_reset = self._last_odom_time >= self._reset_wall_time
                odom_fresh = odom_age < 0.35 and odom_after_reset
                plan_after_reset = self._last_plan_time >= self._reset_wall_time
                threat_ok = self.new_threat
                if step_mode and not threat_ok:
                    threat_ok = (time.time() - self._last_threat_wall_time) < threat_grace_sec
                if threat_ok and odom_fresh and plan_after_reset:
                    self.new_threat = False
                    return True
                remaining = end - time.time()
                if remaining > 0:
                    self.cond.wait(timeout=remaining)
        missing = []
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
        if self._last_plan_time < self._reset_wall_time:
            missing.append("/plan (reset sonrası yeni plan gelmedi)")

        self.node.get_logger().warn(
            f"Obs timeout ({timeout:.1f}s): eksik: {', '.join(missing)}"
        )
        return False

    def _build_goal_state(self) -> np.ndarray:
        with self.cond:
            gx = self._ep_goal_x if self._ep_goal_x is not None else self.goal_x
            gy = self._ep_goal_y if self._ep_goal_y is not None else self.goal_y
            gz = self.goal_z
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
        with self.cond:
            return (self.drone_x, self.drone_y, self.drone_z)

    @staticmethod
    def _astar_forward_start_index(poses, dx: float, dy: float) -> int:
        if not poses:
            return 0
        nearest_idx = 0
        best_dist_sq = float("inf")
        for i, (px, py) in enumerate(poses):
            d_sq = (px - dx) ** 2 + (py - dy) ** 2
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                nearest_idx = i
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
        t = float(np.clip(max_score, 0.0, 1.0))
        if k <= 0.0:
            return 1.0 if t >= threshold else 0.0
        x = k * (t - threshold)
        if x > 40.0:
            return 1.0
        if x < -40.0:
            return 0.0
        return float(1.0 / (1.0 + math.exp(-x)))

    def _nearest_threat_info(self, threat_vec: np.ndarray):
        """Parse object tokens from threat_vector and return nearest valid threat.

        Returns:
            (dist_m, sin_bearing, cos_bearing)  – in drone body frame
            None                                 – no valid object detected

        Token layout per object (THREAT_TOKEN_SIZE = 7):
            [class_id, dist_norm, closing_speed, sin_bear, cos_bear, confidence, is_valid]
        dist_norm is normalised to [0,1]; multiply by LIDAR_MAX_RANGE for metres.
        """
        best_dist_m = float("inf")
        best_sin = 0.0
        best_cos = 1.0
        found = False

        for i in range(self.THREAT_NUM_OBJECTS):
            base = self.THREAT_TOKEN_OFFSET + i * self.THREAT_TOKEN_SIZE
            if base + self.THREAT_TOKEN_SIZE > len(threat_vec):
                break
            if threat_vec[base + self.TK_VALID] < 0.5:
                continue  # invalid / empty slot
            dist_m = float(threat_vec[base + self.TK_DIST_NORM]) * self.LIDAR_MAX_RANGE
            if dist_m < best_dist_m:
                best_dist_m = dist_m
                best_sin   = float(threat_vec[base + self.TK_SIN_BEAR])
                best_cos   = float(threat_vec[base + self.TK_COS_BEAR])
                found = True

        return (best_dist_m, best_sin, best_cos) if found else None



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

    def _pick_episode_goal(self) -> tuple:
        with self.cond:
            real_gx, real_gy = self.goal_x, self.goal_y
            dx, dy = self.drone_x, self.drone_y
            poses = list(self.a_star_poses)

        if real_gx is None:
            return real_gx, real_gy

        direct_dist = math.hypot(real_gx - dx, real_gy - dy)
        if direct_dist <= self.EP_GOAL_MAX_M:
            return real_gx, real_gy   
        if not poses:
            return real_gx, real_gy

        acc = 0.0
        prev_x, prev_y = dx, dy
        for wx, wy in poses:
            seg = math.hypot(wx - prev_x, wy - prev_y)
            if acc + seg >= self.EP_GOAL_MAX_M:
                remain = self.EP_GOAL_MAX_M - acc
                frac = remain / max(seg, 1e-9)
                ix = prev_x + frac * (wx - prev_x)
                iy = prev_y + frac * (wy - prev_y)
                self.node.get_logger().debug(
                    f"EP goal capped at {self.EP_GOAL_MAX_M:.1f}m "
                    f"(real: {direct_dist:.1f}m) → ({ix:.2f}, {iy:.2f})"
                )
                return ix, iy
            acc += seg
            prev_x, prev_y = wx, wy
        return poses[-1]

    def _get_obs(self) -> dict:
        with self.cond:
            lidar_vec = self.threat_vector[self.LIDAR_START_IDX : self.LIDAR_END_IDX].copy()
            threat_vec = self.threat_vector.copy()
            threat_sc = self.threat_scores.copy()
        obs = {
            "lidar_vector": np.nan_to_num(lidar_vec, nan=0.0, posinf=0.0, neginf=0.0),
            "threat_vector": np.nan_to_num(threat_vec, nan=0.0, posinf=0.0, neginf=0.0),
            "threat_scores": np.nan_to_num(np.clip(threat_sc, 0.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0),
            "goal_state": np.nan_to_num(self._build_goal_state(), nan=0.0, posinf=0.0, neginf=0.0),
            "a_star_path": np.nan_to_num(self._build_a_star_obs(), nan=0.0, posinf=0.0, neginf=0.0),
        }
        return obs

    def _dist_to_goal(self) -> float:
        with self.cond:
            gx = self._ep_goal_x if self._ep_goal_x is not None else self.goal_x
            gy = self._ep_goal_y if self._ep_goal_y is not None else self.goal_y
            dx, dy = self.drone_x, self.drone_y
        if gx is None or gy is None:
            return self.MAX_GOAL_DIST
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

    def _check_collision(self, wp_x: float, wp_y: float) -> bool:
        with self.cond:
            dx, dy = self.drone_x, self.drone_y
        if self._check_costmap_collision(dx, dy):
            return True
        if self._check_lidar_collision():
            return True

        stage = self._stage_from_position(self.drone_x) if self.unified_maze else self.curriculum_stage
        if stage in (2, 3) and self._check_actor_collision():
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

    def _lidar_wall_proximity_penalty(self) -> tuple[float, float]:
        """Return (penalty, min_dist_m) from 36-sector lidar for wall proximity shaping."""
        with self.cond:
            lidar = self.threat_vector[self.LIDAR_START_IDX:self.LIDAR_END_IDX].copy()
        if len(lidar) == 0:
            return 0.0, -1.0

        # Ignore empty/invalid sectors to avoid false penalty spikes.
        valid = lidar > 1e-3
        if not np.any(valid):
            return 0.0, -1.0

        min_dist_m = float(np.min(lidar[valid])) * self.LIDAR_MAX_RANGE
        soft = max(self.LIDAR_COLLISION_M + 0.2, self.WALL_SOFT_DIST_M)
        if min_dist_m >= soft:
            return 0.0, min_dist_m

        # Quadratic ramp: smooth far from wall, strong near collision.
        proximity = (soft - min_dist_m) / max(soft - self.LIDAR_COLLISION_M, 1e-6)
        proximity = float(np.clip(proximity, 0.0, 1.0))
        penalty = -self.WALL_PROX_PENALTY_SCALE * (proximity ** 2)
        return float(penalty), min_dist_m

    def _check_actor_collision(self) -> bool:
        if not hasattr(self, 'latest_actor_poses') or not self.latest_actor_poses:
            return False
        if self.drone_z > self.ACTOR_COLLISION_Z_MAX:
            return False

        for ax, ay in self.latest_actor_poses:
            dx = self.drone_x - ax
            dy = self.drone_y - ay
            if math.sqrt(dx * dx + dy * dy) < self.ACTOR_COLLISION_RADIUS:
                return True
        return False
    def _soft_reset_drone(self):
        if self.unified_maze:
            x, y = UNIFIED_ORIGIN
        else:
            x, y = self._stage_origin
        z = self._spawn_z
        dist_to_spawn = math.hypot(self.drone_x - x, self.drone_y - y)
        if dist_to_spawn < 1.0 and abs(self.drone_z - z) < 1.0:
            return 

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

    def _publish_waypoint(self, wx: float, wy: float, wz: float, wyaw: float, is_residual: bool = False):
        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "residual" if is_residual else "odom"
        msg.pose.position.x = wx
        msg.pose.position.y = wy
        msg.pose.position.z = wz
        msg.pose.orientation.z = math.sin(wyaw / 2.0)
        msg.pose.orientation.w = math.cos(wyaw / 2.0)
        self.waypoint_pub.publish(msg)

    def set_sb3_timesteps(self, n: int) -> None:
        try:
            self._sb3_num_timesteps = max(0, int(n))
        except (TypeError, ValueError):
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.smoothed_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_collisions = 0
        self.episode_successes = 0
        self._active_step_count = 0
        self._active_progress_sum = 0.0
        self._active_start_goal_dist = None
        self._active_end_goal_dist = None
        self._active_threat_away_sum = 0.0
        self._active_lateral_escape_sum = 0.0
        with self.cond:
            self.new_threat = False
            self.new_odom = False
            self.threat_vector[:] = 0.0
            self.threat_scores[:] = 0.0
            self.latest_actor_poses = []
            self._last_threat_wall_time = 0.0
        self._last_episode_fatal = False
        
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
            self.node.get_logger().warn("Reset sonrası obs alınamadı, sıfır obs ile devam.")
        ep_gx, ep_gy = self._pick_episode_goal()
        with self.cond:
            self._ep_goal_x = ep_gx
            self._ep_goal_y = ep_gy

        # Reset evasion tracking state at episode start
        with self.cond:
            self._prev_drone_x = self.drone_x
            self._prev_drone_y = self.drone_y
        self._prev_nearest_threat_dist = None
        self._rl_mode = "DORMANT"   # always start dormant; activate only when threat is real

        self.prev_dist_to_goal = self._dist_to_goal()
        if not np.isfinite(self.prev_dist_to_goal):
            self.prev_dist_to_goal = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        raw_action = np.asarray(np.clip(action, -1.0, 1.0), dtype=np.float32)
        effective_action = raw_action.copy()

        # ── DORMANT STATE MACHINE ─────────────────────────────────────────────
        # Risk = weighted combination of threat score + proximity.
        # Evaluated BEFORE EMA so the state machine drives action zeroing.
        with self.cond:
            _sc = self.threat_scores.copy()
            _tv = self.threat_vector.copy()
        _risk_max_threat = float(np.max(_sc))
        _risk_info = self._nearest_threat_info(_tv)
        _risk_dist_m = _risk_info[0] if _risk_info is not None else float("inf")
        _risk_score = (
            self.RL_RISK_W_DIST * max(0.0, 1.0 - _risk_dist_m / self.RL_D_ACTIVATE)
            + self.RL_RISK_W_THREAT * _risk_max_threat
        )
        if self._rl_mode == "DORMANT":
            # Primary activation: risk + geometric proximity.
            # Fallback activation: high threat score even if distance token is absent.
            proximity_trigger = _risk_dist_m < self.RL_D_ACTIVATE
            score_trigger = _risk_max_threat >= self.RL_SCORE_ACTIVATE
            if (
                (_risk_score >= self.RL_RISK_ACTIVATE and proximity_trigger)
                or score_trigger
            ):  
                self._rl_mode = "ACTIVE"
                self.node.get_logger().info(
                    f"[RL] DORMANT→ACTIVE  risk={_risk_score:.2f}  "
                    f"dist={_risk_dist_m:.1f}m  threat={_risk_max_threat:.2f}"
                    f"  (prox={int(proximity_trigger)} score={int(score_trigger)})"
                )
        else:  # ACTIVE
            if (
                _risk_dist_m > self.RL_D_RELEASE
                and _risk_max_threat < self.RL_RISK_RELEASE
            ):
                self._rl_mode = "DORMANT"
                self.node.get_logger().info(
                    f"[RL] ACTIVE→DORMANT  risk={_risk_score:.2f}  "
                    f"dist={_risk_dist_m:.1f}m  threat={_risk_max_threat:.2f}"
                )

        # When dormant: force action to zero so drone doesn't move
        rw_dormant_action_penalty = 0.0
        if self._rl_mode == "DORMANT":
            effective_action = np.zeros(4, dtype=np.float32)
            # Give a small penalty if policy outputs non-zero raw_action in dormant state
            rw_dormant_action_penalty = -0.02 * float(np.sum(np.abs(raw_action)))
        # ────────────────────────────────────────────────────────────

        if self._rl_mode == "ACTIVE":
            ema_alpha = 0.6
        else:
            ema_alpha = self.action_ema_alpha

        self.smoothed_action = (
            ema_alpha * effective_action
            + (1.0 - ema_alpha) * self.smoothed_action
        )

        with self.cond:
            max_threat_pre = float(np.max(self.threat_scores))
            px = self.drone_x

        stage_for_gate = (
            self._stage_from_position(px) if self.unified_maze else self.curriculum_stage
        )

        if stage_for_gate == 1:
            current_step_size = 0.25
            current_yaw_scale = 0.20
            current_path_terminate = max(4.0, self.PATH_ERROR_TERMINATE_M)
            current_path_penalty = -0.5
            current_path_high_threat = 0.3
        elif stage_for_gate == 2:
            current_step_size = 0.45
            current_yaw_scale = 0.15
            current_path_terminate = max(6.0, self.PATH_ERROR_TERMINATE_M)
            current_path_penalty = -0.5
            current_path_high_threat = 0.2
        else:
            current_step_size = 0.60
            current_yaw_scale = 0.25
            current_path_terminate = max(8.0, self.PATH_ERROR_TERMINATE_M)
            current_path_penalty = -0.5
            current_path_high_threat = self.PATH_ERR_SCALE_HIGH_THREAT
        
        if self._rl_mode == "ACTIVE":
            current_step_size *= 2.0
            current_yaw_scale *= 1.5

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
        if stage_for_gate == 1:
            gate_apply = 0.8
        elif stage_for_gate == 2:
            gate_apply = max(0.6, threat_gate)
        else:
            gate_apply = threat_gate if self.THREAT_GATE_ENABLED else 1.0

        dx = float(self.smoothed_action[0]) * current_step_size
        dy = float(self.smoothed_action[1]) * current_step_size
        dz = float(self.smoothed_action[2]) * self.Z_STEP
        dyaw = float(self.smoothed_action[3]) * self.MAX_YAW_RATE

        cos_yaw = math.cos(self.drone_yaw)
        sin_yaw = math.sin(self.drone_yaw)
        world_dx = cos_yaw * dx - sin_yaw * dy
        world_dy = sin_yaw * dx + cos_yaw * dy
        if self._rl_mode == "ACTIVE":
            active_gate_apply = max(0.85, gate_apply)
        else:
            active_gate_apply = gate_apply

        world_dx *= active_gate_apply
        world_dy *= active_gate_apply

        with self.cond:
            poses = list(self.a_star_poses)
            ddx, ddy = self.drone_x, self.drone_y

        # Control authority split:
        # - DORMANT: keep A* hybrid baseline behavior.
        # - ACTIVE: bypass A* anchor entirely so RL has full local control authority.
        # This is required for learning meaningful avoidance maneuvers.
        use_hybrid = (
            self.HYBRID_ASTAR_BASELINE
            and bool(poses)
            and self._rl_mode == "DORMANT"
        )
        if self._rl_mode == "ACTIVE":
            control_source = "RL"
        else:
            control_source = "ASTAR"
        if use_hybrid:
            start = self._astar_forward_start_index(poses, ddx, ddy)
            lx, ly = poses[start]
            wp_x = lx + world_dx
            wp_y = ly + world_dy
            if self.HYBRID_USE_Z_ACTION:
                dz_scaled = float(self.smoothed_action[2]) * self.Z_STEP * active_gate_apply
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
                    self.smoothed_action[3]
                ) * self.MAX_YAW_RATE * current_yaw_scale * active_gate_apply
            else:
                wp_yaw = self.drone_yaw + dyaw * active_gate_apply
            # DORMANT: publish nothing — follow_path falls back to A* within ROUTE_WP_TIMEOUT
            if self._rl_mode == "ACTIVE":
                self._publish_waypoint(wp_x, wp_y, wp_z, wp_yaw, is_residual=True)
        else:
            wp_x = ddx + world_dx
            wp_y = ddy + world_dy
            wp_z = self.drone_z + dz * active_gate_apply
            wp_yaw = self.drone_yaw + dyaw * active_gate_apply
            if self._rl_mode == "ACTIVE":
                self._publish_waypoint(wp_x, wp_y, wp_z, wp_yaw, is_residual=False)

        obs_sync_ok = self._wait_obs(timeout=0.60, step_mode=True)

        if self.unified_maze:
            self.curriculum_stage = self._stage_from_position(self.drone_x)

        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "stage": self.curriculum_stage,
            "obs_sync_ok": obs_sync_ok,
            "threat_gate": threat_gate,
            "threat_gate_applied": active_gate_apply,
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
        rw_wall_proximity = 0.0
        rw_time = 0.0
        rw_stall = 0.0

        reward += rw_dormant_action_penalty

        # ── Progress & stall: only credit RL when it is actively in control ────────
        if self._rl_mode == "ACTIVE" and self.prev_dist_to_goal is not None and self._ep_goal_x is not None:
            progress = self.prev_dist_to_goal - dist
            if not np.isfinite(progress):
                progress = 0.0
            rw_progress = 8.0 * progress
            reward += rw_progress
            if progress < 0.01:
                rw_stall = -0.2
                reward += rw_stall

        if self._rl_mode == "ACTIVE":
            self._active_step_count += 1
            self._active_progress_sum += float(rw_progress)

            if self._active_start_goal_dist is None:
                self._active_start_goal_dist = float(self.prev_dist_to_goal if self.prev_dist_to_goal is not None else dist)
            self._active_end_goal_dist = float(dist)
        # ── Collision & goal: always apply (safety + terminal signal) ──────────
        if self._check_collision(wp_x, wp_y):
            rw_collision = -20.0
            reward += rw_collision
            terminated = True
            info["collision"] = True
            self.episode_collisions += 1
            self._last_episode_fatal = True
        elif dist < self.GOAL_TOLERANCE:
            rw_goal = 50.0
            reward += rw_goal
            terminated = True
            info["success"] = True
            self.episode_successes += 1

        # ── Path tracking: only in ACTIVE mode ──────────────────────────────
        path_err = self._dist_to_nearest_astar_wp()
        path_err_normalized = min(path_err / 5.0, 1.0)
        on_astar = path_err < self.ASTAR_ON_PATH_M
        info["on_astar"] = on_astar
        if self._rl_mode == "ACTIVE":
            if self.THREAT_GATE_ENABLED:
                path_w = (
                    self.PATH_ERR_SCALE_LOW_THREAT * (1.0 - threat_gate)
                    + current_path_high_threat * threat_gate
                )
                rw_path = current_path_penalty * path_err_normalized * path_w
                reward += rw_path
                if (
                    threat_gate < self.ASTAR_RETURN_GATE_MAX
                    and path_err < self.ASTAR_ON_PATH_M
                ):
                    rw_astar_return = self.ASTAR_RETURN_BONUS
                    reward += rw_astar_return
            else:
                rw_path = current_path_penalty * path_err_normalized
                reward += rw_path

        # ── Path-drift termination: always (safety) ─────────────────────────
        if not terminated and current_path_terminate > 0.0:
            with self.cond:
                has_plan = len(self.a_star_poses) >= 2
            if has_plan and path_err > current_path_terminate:
                rw_path_drift = self.PATH_ERROR_TERMINATE_PENALTY
                reward += rw_path_drift
                terminated = True
                info["path_drift_terminate"] = True

        # ── Threat penalty: always (awareness must exist in both modes) ────────
        max_threat = float(np.max(self.threat_scores))
        if stage_for_gate == 1:
            threat_penalty_scale = -0.5
        elif stage_for_gate == 2:
            threat_penalty_scale = -1.0
        else:
            threat_penalty_scale = -2.0
        rw_threat = threat_penalty_scale * max_threat
        reward += rw_threat

        # ── Wall proximity: ACTIVE-mode shaping from 36-sector lidar ─────────
        min_lidar_wall_dist_m = -1.0
        if self._rl_mode == "ACTIVE":
            rw_wall_proximity, min_lidar_wall_dist_m = self._lidar_wall_proximity_penalty()
            reward += rw_wall_proximity

        # ── Evasion rewards: only when ACTIVE and threat is real/nearby ───────
        rw_threat_away    = 0.0
        rw_lateral_escape = 0.0
        nearest_threat_dist = float("inf")

        with self.cond:
            tv = self.threat_vector.copy()
            cur_dx, cur_dy = self.drone_x, self.drone_y
            cur_yaw = self.drone_yaw

        threat_info = self._nearest_threat_info(tv)

        if threat_info is not None:
            nearest_threat_dist, sin_bear, cos_bear = threat_info
            evasion_active = (
                self._rl_mode == "ACTIVE"
                and max_threat >= self.EVASION_THREAT_GATE
                and nearest_threat_dist <= self.EVASION_DIST_MAX_M
            )
            if evasion_active:
                if self._prev_nearest_threat_dist is not None:
                    away_delta = nearest_threat_dist - self._prev_nearest_threat_dist
                    if away_delta > 0.0:
                        rw_threat_away = self.THREAT_AWAY_SCALE * away_delta
                        reward += rw_threat_away
                        self._active_threat_away_sum += float(rw_threat_away)
                move_x = cur_dx - self._prev_drone_x
                move_y = cur_dy - self._prev_drone_y
                move_speed = math.hypot(move_x, move_y)
                if move_speed >= self.LATERAL_MIN_MOVE_M:
                    bearing = math.atan2(sin_bear, cos_bear)
                    world_angle = cur_yaw + bearing
                    threat_dir_x = math.cos(world_angle)
                    threat_dir_y = math.sin(world_angle)
                    lateral_m = abs(move_x * threat_dir_y - move_y * threat_dir_x)
                    lateral_frac = lateral_m / max(move_speed, 1e-6)
                    rw_lateral_escape = self.LATERAL_ESCAPE_SCALE * lateral_frac * move_speed
                    reward += rw_lateral_escape
                    self._active_lateral_escape_sum += float(rw_lateral_escape)

        self._prev_drone_x = cur_dx
        self._prev_drone_y = cur_dy
        self._prev_nearest_threat_dist = nearest_threat_dist if threat_info is not None else None

        reward += rw_time

        if self.step_count >= self.MAX_EPISODE_STEPS:
            truncated = True
            info["timeout"] = True

        self.prev_dist_to_goal = dist
        self.prev_action = effective_action.copy()
        self.episode_reward += reward

        # ── Publish dormant-mode debug topics ──────────────────────────────
        _mode_msg = String()
        _mode_msg.data = self._rl_mode
        self._rl_mode_pub.publish(_mode_msg)
        _risk_msg = Float32()
        _risk_msg.data = float(_risk_score)
        self._rl_risk_pub.publish(_risk_msg)

        info["path_error"] = path_err
        info["max_threat"] = max_threat
        info["dist_to_goal"] = dist
        info["episode_reward"] = self.episode_reward
        info["episode_collisions"] = self.episode_collisions
        info["episode_successes"] = self.episode_successes
        info["rl_mode"] = self._rl_mode
        info["rl_risk_score"] = float(_risk_score)
        info["rl_risk_dist_m"] = float(_risk_dist_m) if np.isfinite(_risk_dist_m) else -1.0
        info["rl_risk_max_threat"] = float(_risk_max_threat)
        info["control_source"] = control_source
        info["rw_progress"] = rw_progress
        info["rw_goal"] = rw_goal
        info["rw_collision"] = rw_collision
        info["rw_path"] = rw_path
        info["rw_stall"] = rw_stall
        info["rw_astar_return"] = rw_astar_return
        info["rw_path_drift"] = rw_path_drift
        info["rw_threat"] = rw_threat
        info["rw_wall_proximity"] = rw_wall_proximity
        info["rw_time"] = rw_time
        info["rw_dormant_action_penalty"] = rw_dormant_action_penalty
        info["rw_threat_away"]    = rw_threat_away
        info["rw_lateral_escape"] = rw_lateral_escape
        info["nearest_threat_dist"] = nearest_threat_dist if nearest_threat_dist < float("inf") else -1.0
        info["min_lidar_wall_dist_m"] = float(min_lidar_wall_dist_m)
        info["raw_action_norm"] = float(np.linalg.norm(raw_action))
        info["effective_action_norm"] = float(np.linalg.norm(effective_action))
        info["active_step_count"] = int(self._active_step_count)
        info["active_progress_sum"] = float(self._active_progress_sum)
        info["active_goal_dist_start"] = (
            float(self._active_start_goal_dist)
            if self._active_start_goal_dist is not None else -1.0
        )
        info["active_goal_dist_end"] = (
            float(self._active_end_goal_dist)
            if self._active_end_goal_dist is not None else -1.0
        )
        info["active_goal_dist_delta"] = (
            float(self._active_start_goal_dist - self._active_end_goal_dist)
            if self._active_start_goal_dist is not None and self._active_end_goal_dist is not None
            else 0.0
        )
        info["active_threat_away_sum"] = float(self._active_threat_away_sum)
        info["active_lateral_escape_sum"] = float(self._active_lateral_escape_sum)

        obs = self._get_obs()
        for k, v in obs.items():
            if not np.all(np.isfinite(v)):
                self.node.get_logger().warn(f"Non-finite observation in {k}, zeroing it")
                obs[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(reward):
            self.node.get_logger().warn(f"Non-finite reward detected: {reward}, forcing to 0.0")
            reward = 0.0
        return obs, float(reward), terminated, truncated, info


    def close(self):
        self._running = False
        time.sleep(0.2)
        if self.node:
            self.node.destroy_node()
        if getattr(self, "_rclpy_initialized", False) and rclpy.ok():
            rclpy.shutdown()
            