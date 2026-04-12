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
    NUM_PATH_WPS = 5

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

        with self.cond:
            self.new_threat = False
            self.new_odom = False
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

        with self.cond:
            self._ep_goal_x = self.goal_x
            self._ep_goal_y = self.goal_y

        self.prev_dist_to_goal = self._dist_to_goal()
        if not np.isfinite(self.prev_dist_to_goal):
            self.prev_dist_to_goal = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)
        self.smoothed_action = self.action_ema_alpha * action + (1.0 - self.action_ema_alpha) * self.smoothed_action

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
            current_step_size = 0.25         
            current_yaw_scale = 0.15
            current_path_terminate = max(6.0, self.PATH_ERROR_TERMINATE_M)
            current_path_penalty = -2.0
            current_path_high_threat = 0.2
        else: 
            current_step_size = 0.30
            current_yaw_scale = 0.25
            current_path_terminate = max(8.0, self.PATH_ERROR_TERMINATE_M)
            current_path_penalty = -0.5
            current_path_high_threat = self.PATH_ERR_SCALE_HIGH_THREAT

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
                dz_scaled = float(self.smoothed_action[2]) * self.Z_STEP * gate_apply
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
                ) * self.MAX_YAW_RATE * current_yaw_scale * gate_apply
            else:
                wp_yaw = self.drone_yaw + dyaw * gate_apply
                
            self._publish_waypoint(wp_x, wp_y, wp_z, wp_yaw, is_residual=True)
        else:
            wp_x = ddx + world_dx
            wp_y = ddy + world_dy
            wp_z = self.drone_z + dz * gate_apply
            wp_yaw = self.drone_yaw + dyaw * gate_apply

            self._publish_waypoint(wp_x, wp_y, wp_z, wp_yaw, is_residual=False)
        obs_sync_ok = self._wait_obs(timeout=0.40, step_mode=True)

        if self.unified_maze:
            self.curriculum_stage = self._stage_from_position(self.drone_x)

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
        rw_time = 0.0   
        rw_stall = 0.0
        if self.prev_dist_to_goal is not None and self._ep_goal_x is not None:  
            progress = self.prev_dist_to_goal - dist

            if not np.isfinite(progress):
                progress = 0.0

            rw_progress = 8.0 * progress
            reward += rw_progress

            if progress < 0.01:
                rw_stall = -0.2
                reward += rw_stall

        if self._check_collision(wp_x, wp_y):
            rw_collision = -100.0
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

        path_err = self._dist_to_nearest_astar_wp()
        path_err_normalized = min(path_err / 5.0, 1.0)
        on_astar = path_err < self.ASTAR_ON_PATH_M
        info["on_astar"] = on_astar
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

        if not terminated and current_path_terminate > 0.0:
            with self.cond:
                has_plan = len(self.a_star_poses) >= 2
            if has_plan and path_err > current_path_terminate:
                rw_path_drift = self.PATH_ERROR_TERMINATE_PENALTY
                reward += rw_path_drift
                terminated = True
                info["path_drift_terminate"] = True

        max_threat = float(np.max(self.threat_scores))
        if stage_for_gate == 1:
            threat_penalty_scale = -0.5
        elif stage_for_gate == 2:
            threat_penalty_scale = -1.0
        else:
            threat_penalty_scale = -2.0
        rw_threat = threat_penalty_scale * max_threat
        reward += rw_threat

        reward += rw_time

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
        info["rw_stall"] = rw_stall
        info["rw_astar_return"] = rw_astar_return
        info["rw_path_drift"] = rw_path_drift
        info["rw_threat"] = rw_threat
        info["rw_time"] = rw_time

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
            