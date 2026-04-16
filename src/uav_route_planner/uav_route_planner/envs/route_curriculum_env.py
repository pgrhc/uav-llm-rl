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

from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry, OccupancyGrid
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
    STEP_SIZE = float(os.environ.get("ROUTE_STEP_SIZE", "1.20"))
    Z_STEP = 0.2
    MAX_YAW_RATE = 0.52
    GOAL_TOLERANCE = 1.3
    SAFETY_RADIUS = 0.5
    LETHAL_THRESHOLD = 90
    MAX_EPISODE_STEPS = 2048
    MAX_GOAL_DIST = 30.0
    MAX_SPEED = 5.0

    LIDAR_MAX_RANGE = 30.0
    LIDAR_COLLISION_M = 0.60
    LIDAR_START_IDX = 3
    LIDAR_END_IDX = 39
    # ── Reward thresholds ────────────────────────────────────────────────────
    # Lidar < WARN_M → "too close" → −5 penalty each step
    # Lidar >= WARN_M → "safe"      → +1 bonus each step
    LIDAR_WARN_M = 2.0

    NUM_PATH_WPS = 5

    # ── Threat token layout (threat_vector[39:74] = 5 × 7) ──────────────────
    THREAT_TOKEN_OFFSET  = 39
    THREAT_TOKEN_SIZE    = 7
    THREAT_NUM_OBJECTS   = 5
    TK_CLASS_ID   = 0
    TK_DIST_NORM  = 1
    TK_CLOSING    = 2
    TK_SIN_BEAR   = 3
    TK_COS_BEAR   = 4
    TK_CONFIDENCE = 5
    TK_VALID      = 6

    DRONE_MODEL_NAME = "x500_mono_cam_0"
    GZ_WORLD_NAME = "default"
    RESET_STABILIZE_SEC = 2.0
    CRUISE_Z = float(os.environ.get("ROUTE_CRUISE_Z", "1.5"))
    EP_GOAL_MAX_M = float(os.environ.get("ROUTE_EP_GOAL_MAX_M", "5.0"))

    ACTOR_COLLISION_RADIUS = 1.5
    ACTOR_COLLISION_Z_MAX = 3.5

    # ── Simplified reward constants ──────────────────────────────────────────
    # +2   progress toward goal (scaled by distance delta)
    # +1   safe from obstacles (min lidar >= LIDAR_WARN_M)
    # +100 goal reached
    # −5   flying too close to obstacles (min lidar < LIDAR_WARN_M)
    # −0.1 per-step time penalty
    # −100 collision (costmap / actor / lidar hit)
    RW_PROGRESS_SCALE    = float(os.environ.get("ROUTE_RW_PROGRESS_SCALE",    "1.5"))
    RW_SAFE_BONUS        = float(os.environ.get("ROUTE_RW_SAFE_BONUS",        "1.0"))
    RW_GOAL              = float(os.environ.get("ROUTE_RW_GOAL",              "100.0"))
    RW_TOO_CLOSE_PENALTY = float(os.environ.get("ROUTE_RW_TOO_CLOSE_PENALTY", "-2.0"))
    RW_TIME_PENALTY      = float(os.environ.get("ROUTE_RW_TIME_PENALTY",      "-0.1"))
    RW_COLLISION_PENALTY = float(os.environ.get("ROUTE_RW_COLLISION_PENALTY", "-20.0"))
    RW_SHIELD_PENALTY = float(os.environ.get("ROUTE_RW_SHIELD_PENALTY", "-8.0"))
    RW_SHIELD_BACKOFF = float(os.environ.get("ROUTE_RW_SHIELD_BACKOFF", "-1.0"))
    SHIELD_MIN_SCALE = float(os.environ.get("ROUTE_SHIELD_MIN_SCALE", "0.15"))
    SHIELD_SCALE_STEPS = int(os.environ.get("ROUTE_SHIELD_SCALE_STEPS", "6"))
    RW_PROGRESS_STEP_REWARD = float(os.environ.get("ROUTE_RW_PROGRESS_STEP_REWARD", "2.0"))


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
            "lidar_vector": spaces.Box(
                low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32
            ),
            "threat_vector": spaces.Box(
                low=-np.inf, high=np.inf, shape=(74,), dtype=np.float32
            ),
            "threat_scores": spaces.Box(
                low=0.0, high=1.0, shape=(5,), dtype=np.float32
            ),
            "goal_state": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
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
        self.action_ema_alpha = float(os.environ.get("ROUTE_ACTION_EMA_ALPHA", "1.0"))
        self.step_count = 0

        self.threat_vector = np.zeros(74, dtype=np.float32)
        self.threat_scores = np.zeros(5, dtype=np.float32)
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

        self._sb3_num_timesteps = 0

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
            PoseArray, "/route/actor_poses", self._cb_actor_poses, 10)

        self.waypoint_pub = self.node.create_publisher(
            PoseStamped, "/route/waypoint_desired", 10)
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

    def _segment_is_safe(self, x0: float, y0: float, x1: float, y1: float, samples: int = 12) -> bool:
        for i in range(1, samples + 1):
            t = i / samples
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            if self._check_costmap_collision(x, y):
                return False
        return True
    
    def _apply_waypoint_shield(self, x0: float, y0: float, z0: float,
                           x1: float, y1: float, z1: float):
        if self._segment_is_safe(x0, y0, x1, y1):
            return x1, y1, z1, False, False, 1.0

        # Aynı yön üzerinde kademeli küçültme dene
        for scale in np.linspace(0.8, self.SHIELD_MIN_SCALE, self.SHIELD_SCALE_STEPS):
            sx = x0 + scale * (x1 - x0)
            sy = y0 + scale * (y1 - y0)
            sz = z0 + scale * (z1 - z0)
            if self._segment_is_safe(x0, y0, sx, sy):
                return sx, sy, sz, True, False, float(scale)

        # Hâlâ güvenli değilse hafif geri çek
        vx = x1 - x0
        vy = y1 - y0
        norm = math.hypot(vx, vy)

        if norm > 1e-6:
            backoff_dist = min(0.25, 0.35 * norm)
            bx = x0 - backoff_dist * (vx / norm)
            by = y0 - backoff_dist * (vy / norm)
            bz = z0
            if self._segment_is_safe(x0, y0, bx, by):
                return bx, by, bz, True, True, -backoff_dist

        # Son çare: yerinde kal
        return x0, y0, z0, True, False, 0.0

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
            self.goal_x = msg.pose.position.x
            self.goal_y = msg.pose.position.y
            self.goal_z = msg.pose.position.z

    def _cb_costmap(self, msg: OccupancyGrid):
        with self.cond:
            self.latest_costmap = msg

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
                threat_ok = self.new_threat
                if step_mode and not threat_ok:
                    threat_ok = (time.time() - self._last_threat_wall_time) < threat_grace_sec
                if threat_ok and odom_fresh:
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
            dyaw = 0.0
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
            rel_x / self.MAX_GOAL_DIST,   # normalize et
            rel_y / self.MAX_GOAL_DIST,   # normalize et
            np.clip(rel_z / 5.0, -1.0, 1.0),
            dist_norm,
            speed_norm,
            math.sin(dyaw),
            math.cos(dyaw),
        ], dtype=np.float32)

    def get_drone_position(self):
        with self.cond:
            return (self.drone_x, self.drone_y, self.drone_z)

    def _nearest_threat_info(self, threat_vec: np.ndarray):
        best_dist_m = float("inf")
        best_sin = 0.0
        best_cos = 1.0
        found = False
        for i in range(self.THREAT_NUM_OBJECTS):
            base = self.THREAT_TOKEN_OFFSET + i * self.THREAT_TOKEN_SIZE
            if base + self.THREAT_TOKEN_SIZE > len(threat_vec):
                break
            if threat_vec[base + self.TK_VALID] < 0.5:
                continue
            dist_m = float(threat_vec[base + self.TK_DIST_NORM]) * self.LIDAR_MAX_RANGE
            if dist_m < best_dist_m:
                best_dist_m = dist_m
                best_sin   = float(threat_vec[base + self.TK_SIN_BEAR])
                best_cos   = float(threat_vec[base + self.TK_COS_BEAR])
                found = True
        return (best_dist_m, best_sin, best_cos) if found else None

    def _pick_episode_goal(self) -> tuple:
        with self.cond:
            dx, dy = self.drone_x, self.drone_y

        if self.goal_x is not None and self.goal_y is not None:
            return self.goal_x, self.goal_y

        sx, sy = self._stage_origin
        candidates = [
            (sx + 8.0, sy + 0.0),
            (sx - 8.0, sy + 0.0),
            (sx + 0.0, sy + 8.0),
            (sx + 0.0, sy - 8.0),
            (sx + 10.0, sy + 10.0),
            (sx - 10.0, sy - 10.0),
        ]
        filtered = [(gx, gy) for gx, gy in candidates if math.hypot(gx - dx, gy - dy) >= 5.0]
        if not filtered:
            filtered = candidates
        idx = int(self.np_random.integers(0, len(filtered)))
        return filtered[idx]

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
        """Returns True if any valid lidar sector reads < LIDAR_COLLISION_M."""
        lidar = self.threat_vector[self.LIDAR_START_IDX:self.LIDAR_END_IDX]
        if len(lidar) == 0:
            return False
        valid = lidar > 1e-3
        if not np.any(valid):
            return False
        min_dist_m = float(np.min(lidar[valid])) * self.LIDAR_MAX_RANGE
        return min_dist_m < self.LIDAR_COLLISION_M

    def _min_lidar_dist_m(self) -> float:
        """Returns the minimum valid lidar reading in metres, or inf if none."""
        lidar = self.threat_vector[self.LIDAR_START_IDX:self.LIDAR_END_IDX]
        if len(lidar) == 0:
            return float("inf")
        valid = lidar > 1e-3
        if not np.any(valid):
            return float("inf")
        return float(np.min(lidar[valid])) * self.LIDAR_MAX_RANGE

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
        # reset() içinde:
        self._prev_min_lidar_m = float("inf")
        with self.cond:
            self.new_threat = False
            self.new_odom = False
            self.threat_vector[:] = 0.0
            self.threat_scores[:] = 0.0
            self.latest_actor_poses = []
            self._last_threat_wall_time = 0.0

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

        self.prev_dist_to_goal = self._dist_to_goal()
        if not np.isfinite(self.prev_dist_to_goal):
            self.prev_dist_to_goal = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        raw_action = np.asarray(np.clip(action, -1.0, 1.0), dtype=np.float32)
        self.smoothed_action = (
            self.action_ema_alpha * raw_action
            + (1.0 - self.action_ema_alpha) * self.smoothed_action
        )
        current_step_size = 0.80

        dx = float(self.smoothed_action[0]) * current_step_size
        dy = float(self.smoothed_action[1]) * current_step_size
        dz = float(self.smoothed_action[2]) * self.Z_STEP
        dyaw = float(self.smoothed_action[3]) * self.MAX_YAW_RATE

        with self.cond:
            ddx, ddy, ddz = self.drone_x, self.drone_y, self.drone_z
            dyaw_world = self.drone_yaw

        wp_x_raw = ddx + dx
        wp_y_raw = ddy + dy
        wp_z_raw = ddz + dz
        wp_yaw = dyaw_world + dyaw

        wp_x, wp_y, wp_z, shield_used, backoff_used, applied_scale = self._apply_waypoint_shield(
            ddx, ddy, ddz,
            wp_x_raw, wp_y_raw, wp_z_raw
        )

        self._publish_waypoint(wp_x, wp_y, wp_z, wp_yaw, is_residual=False)

        obs_sync_ok = self._wait_obs(timeout=0.60, step_mode=True)

        if self.unified_maze:
            self.curriculum_stage = self._stage_from_position(self.drone_x)

        terminated = False
        truncated = False
        info = {
            "stage": self.curriculum_stage,
            "obs_sync_ok": obs_sync_ok,
            "control_source": "RL",
            "shield_used": shield_used,
            "shield_backoff": backoff_used,
            "shield_scale": float(applied_scale),
        }

        dist = self._dist_to_goal()

        # ── Lidar distance (used for safe/too-close decisions) ─────────────
        min_lidar_m = self._min_lidar_dist_m()

        # ── Collision detection ────────────────────────────────────────────
        #   Priority order: costmap > actor > lidar hard-limit > goal > obstacle
        #
        #   All three collision types now terminate the episode.
        #   This avoids the PX4 internal state divergence that occurs when we
        #   only do a Gazebo pose reset without also resetting PX4's EKF.
        with self.cond:
            drone_x_now = self.drone_x
            drone_y_now = self.drone_y

        stage_cur = (
            self._stage_from_position(drone_x_now)
            if self.unified_maze
            else self.curriculum_stage
        )

        costmap_hit = self._check_costmap_collision(drone_x_now, drone_y_now)
        actor_hit   = stage_cur in (2, 3) and self._check_actor_collision()
        lidar_hit   = self._check_lidar_collision()

        # ── Simplified reward ──────────────────────────────────────────────
        #   +2   progress toward goal   (scaled by metres closed per step)
        #   +1   safe from obstacles    (min lidar >= LIDAR_WARN_M)
        #   +100 goal reached
        #   −5   flying too close       (min lidar < LIDAR_WARN_M)
        #   −0.1 per-step time penalty
        #   −100 any collision

        reward = self.RW_TIME_PENALTY  # −0.1 every step
        if shield_used:
            if applied_scale >= 0.0:
                correction_ratio = 1.0 - applied_scale
            else:
                correction_ratio = 1.0

            reward += self.RW_SHIELD_PENALTY * (0.5 + correction_ratio)
            info["shield_penalty_applied"] = True
            info["shield_correction_ratio"] = float(correction_ratio)

            if backoff_used:
                reward += self.RW_SHIELD_BACKOFF

            # terminated = True
            # info["collision"] = True
            # info["collision_type"] = "shield"
            # self.episode_collisions += 1
        elif costmap_hit or actor_hit or lidar_hit:
            # ── Collision: terminate immediately, heavy penalty ────────────
            reward += self.RW_COLLISION_PENALTY   # −100
            terminated = True
            info["collision"] = True
            if costmap_hit:
                info["collision_type"] = "costmap"
            elif actor_hit:
                info["collision_type"] = "actor"
            else:
                info["collision_type"] = "lidar"
            self.episode_collisions += 1
            self.node.get_logger().warn(
                f"[Collision] type={info['collision_type']} "
                f"lidar_min={min_lidar_m:.2f}m"
            )

        elif dist < self.GOAL_TOLERANCE:
            # ── Goal reached ──────────────────────────────────────────────
            reward += self.RW_GOAL   # +100
            terminated = True
            info["success"] = True
            self.episode_successes += 1

        else:
            # ── Progress (potansiyel bazlı) ────────────────────────────────
            if self.prev_dist_to_goal is not None:
                progress = self.prev_dist_to_goal - dist
                reward += self.RW_PROGRESS_SCALE * progress   # 2.0 * progress
                if np.isfinite(progress):
                    info["progress"] = float(progress)

            # ── Yön bonusu: goal'a doğru mu bakıyor? ──────────────────────
            # Mevcut hareketi goal yönüyle karşılaştır
            with self.cond:
                gx = self._ep_goal_x
                gy = self._ep_goal_y
                px, py = self.drone_x, self.drone_y
            # if gx is not None:
            #     goal_dir_x = gx - px
            #     goal_dir_y = gy - py
            #     goal_norm = math.hypot(goal_dir_x, goal_dir_y)
            #     if goal_norm > 0.1:
            #         goal_dir_x /= goal_norm
            #         goal_dir_y /= goal_norm
            #         # action dx, dy zaten step() içinde hesaplandı
            #         heading_dot = dx * goal_dir_x + dy * goal_dir_y
            #         # [-1, 1] arası → goal yönünde hareket ediyorsa pozitif
            #         reward += 0.3 * heading_dot
            #         info["heading_dot"] = float(heading_dot)

            # ── Engel mesafesi ─────────────────────────────────────────────
            if min_lidar_m < self.LIDAR_WARN_M:
                reward += self.RW_TOO_CLOSE_PENALTY   # −2.0
                info["too_close"] = True
                if hasattr(self, '_prev_min_lidar_m'):
                    escape_delta = min_lidar_m - self._prev_min_lidar_m
                    if escape_delta > 0.05:
                        reward += 0.5 * escape_delta
                        info["wall_escape_bonus"] = True
            else:
                reward += 0.02 
                info["too_close"] = False

            self._prev_min_lidar_m = min_lidar_m

        if self.step_count >= self.MAX_EPISODE_STEPS:
            truncated = True
            info["timeout"] = True

        self.prev_dist_to_goal = dist
        self.prev_action = raw_action.copy()
        self.episode_reward += reward

        info["dist_to_goal"]        = dist
        info["min_lidar_m"]         = float(min_lidar_m)
        info["episode_reward"]      = self.episode_reward
        info["episode_collisions"]  = self.episode_collisions
        info["episode_successes"]   = self.episode_successes
        info["raw_action_norm"]     = float(np.linalg.norm(raw_action))

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