#!/usr/bin/env python3
"""
Route Planning Agent — Curriculum Learning Environment

Observation (same as RouteEnv):
    costmap_patch  : (1, 64, 64)  — CNN   /route/costmap_patch
    threat_vector  : (74,)        — MLP   /threat/state_vec
    threat_scores  : (5,)         — MLP   /threat/target_scores
    goal_state     : (7,)         — MLP   (rel_goal, dist, speed, yaw)
    a_star_path    : (10,)        — MLP   /plan  (5 wp × rel_x,rel_y)

Action:
    Box(4,) → (dx, dy, dz, dyaw)

Reward (FIXED across all stages — components naturally activate):
    r_progress          +2.0 × (prev_dist − curr_dist)
    r_goal              +50.0
    r_collision          −100.0   (costmap lethal / LiDAR / actor)
    r_path_error        −1.0 × dist_to_nearest_astar_wp
    r_threat_proximity  −2.0 × max(threat_scores)
    r_smooth            −0.3 × ‖Δaction‖
    r_time              −0.05

Stages:
    1  Path following — open maze, no actors
    2  Static obstacles — narrow corridors, dead-ends
    3  Dynamic threats — 180-200 walking actors
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
SECTION1_X_MAX = 50.0
SECTION2_X_MAX = 125.0


class RouteCurriculumEnv(gym.Env):
    metadata = {"render_modes": []}

    STEP_SIZE = 0.3
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
    RESET_STABILIZE_SEC = 5.0  # Teleport sonrasi EKF stabilizasyonu icin

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

        self.new_patch = False
        self.new_threat = False
        self.new_odom = False
        self.cond = threading.Condition()

        self.bridge = CvBridge()

        # Actor tracking (Stage 3)
        self.actor_trajectories = []
        self.actor_ref_time = time.time()

        # Episode stats
        self.episode_reward = 0.0
        self.episode_collisions = 0
        self.episode_successes = 0

        # --- ROS 2 ---
        if not rclpy.ok():
            rclpy.init()

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
            OccupancyGrid, "/local_costmap/costmap", self._cb_costmap, qos_reliable)
        self.node.create_subscription(
            Path, "/plan", self._cb_plan, qos_sensor)

        self.waypoint_pub = self.node.create_publisher(
            PoseStamped, "/route/waypoint_desired", 10)

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        time.sleep(1.0)

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
            rclpy.spin_once(self.node, timeout_sec=0.1)

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
            self.cond.notify_all()

    def _cb_goal(self, msg: PoseStamped):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.goal_z = msg.pose.position.z

    def _cb_costmap(self, msg: OccupancyGrid):
        self.latest_costmap = msg

    def _cb_plan(self, msg: Path):
        self.a_star_poses = [
            (p.pose.position.x, p.pose.position.y) for p in msg.poses
        ]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _wait_obs(self, timeout: float = 0.5) -> bool:
        end = time.time() + timeout
        with self.cond:
            while time.time() < end:
                if self.new_patch and self.new_threat and self.new_odom:
                    self.new_patch = False
                    self.new_threat = False
                    self.new_odom = False
                    return True
                remaining = end - time.time()
                if remaining > 0:
                    self.cond.wait(timeout=remaining)
        return False

    def _build_goal_state(self) -> np.ndarray:
        if self.goal_x is None:
            return np.zeros(7, dtype=np.float32)

        rel_x = self.goal_x - self.drone_x
        rel_y = self.goal_y - self.drone_y
        rel_z = (self.goal_z if self.goal_z else self.drone_z) - self.drone_z
        dist = math.sqrt(rel_x * rel_x + rel_y * rel_y)
        dist_norm = min(dist / self.MAX_GOAL_DIST, 1.0)
        speed_norm = min(self.drone_speed / self.MAX_SPEED, 1.0)

        return np.array([
            rel_x, rel_y, rel_z,
            dist_norm, speed_norm,
            math.sin(self.drone_yaw),
            math.cos(self.drone_yaw),
        ], dtype=np.float32)

    def _build_a_star_obs(self) -> np.ndarray:
        result = np.zeros(self.NUM_PATH_WPS * 2, dtype=np.float32)
        if not self.a_star_poses:
            return result

        dx = self.drone_x
        dy = self.drone_y

        nearest_idx = 0
        best_dist = float("inf")
        for i, (px, py) in enumerate(self.a_star_poses):
            d = (px - dx) ** 2 + (py - dy) ** 2
            if d < best_dist:
                best_dist = d
                nearest_idx = i

        start = nearest_idx + 1
        count = 0
        for i in range(start, len(self.a_star_poses)):
            if count >= self.NUM_PATH_WPS:
                break
            wx, wy = self.a_star_poses[i]
            result[count * 2] = wx - dx
            result[count * 2 + 1] = wy - dy
            count += 1

        return result

    def _get_obs(self) -> dict:
        return {
            "costmap_patch": self.costmap_patch.copy(),
            "threat_vector": self.threat_vector.copy(),
            "threat_scores": np.clip(self.threat_scores.copy(), 0.0, 1.0),
            "goal_state": self._build_goal_state(),
            "a_star_path": self._build_a_star_obs(),
        }

    def _dist_to_goal(self) -> float:
        if self.goal_x is None:
            return float("inf")
        dx = self.goal_x - self.drone_x
        dy = self.goal_y - self.drone_y
        return math.sqrt(dx * dx + dy * dy)

    def _dist_to_nearest_astar_wp(self) -> float:
        if not self.a_star_poses:
            return 0.0
        best = float("inf")
        for px, py in self.a_star_poses:
            d = math.sqrt((px - self.drone_x) ** 2 + (py - self.drone_y) ** 2)
            if d < best:
                best = d
        return best

    # ------------------------------------------------------------------ #
    # Collision detection
    # ------------------------------------------------------------------ #
    def _check_collision(self, wp_x: float, wp_y: float) -> bool:
        if self._check_costmap_collision(wp_x, wp_y):
            return True
        if self._check_lidar_collision():
            return True
        stage = self._stage_from_position(self.drone_x) if self.unified_maze else self.curriculum_stage
        if stage == 3 and self._check_actor_collision():
            return True
        return False

    def _check_costmap_collision(self, x: float, y: float) -> bool:
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

    # ------------------------------------------------------------------ #
    # Teleportation / soft reset
    # ------------------------------------------------------------------ #
    def _soft_reset_drone(self):
        if self.unified_maze:
            x, y = UNIFIED_ORIGIN
        else:
            x, y = self._stage_origin
        z = self._spawn_z
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
            subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3)
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
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.episode_reward = 0.0

        self._soft_reset_drone()
        self._wait_obs(timeout=2.0)

        self.prev_dist_to_goal = self._dist_to_goal()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        dx = float(action[0]) * self.STEP_SIZE
        dy = float(action[1]) * self.STEP_SIZE
        dz = float(action[2]) * self.Z_STEP
        dyaw = float(action[3]) * self.MAX_YAW_RATE

        cos_yaw = math.cos(self.drone_yaw)
        sin_yaw = math.sin(self.drone_yaw)
        world_dx = cos_yaw * dx - sin_yaw * dy
        world_dy = sin_yaw * dx + cos_yaw * dy

        wp_x = self.drone_x + world_dx
        wp_y = self.drone_y + world_dy
        wp_z = self.drone_z + dz
        wp_yaw = self.drone_yaw + dyaw

        self._publish_waypoint(wp_x, wp_y, wp_z, wp_yaw)
        self._wait_obs(timeout=0.3)

        # Birlesik maze: stage pozisyondan hesapla
        if self.unified_maze:
            self.curriculum_stage = self._stage_from_position(self.drone_x)

        # ══════════════════════════════════════════════════════════════
        # FIXED REWARD (identical formula across all stages)
        # ══════════════════════════════════════════════════════════════
        reward = 0.0
        terminated = False
        truncated = False
        info = {"stage": self.curriculum_stage}

        dist = self._dist_to_goal()

        # r_progress
        if self.prev_dist_to_goal is not None and self.goal_x is not None:
            progress = self.prev_dist_to_goal - dist
            reward += 2.0 * progress

        # r_goal
        if dist < self.GOAL_TOLERANCE:
            reward += 50.0
            terminated = True
            info["success"] = True

        # r_collision
        if self._check_collision(wp_x, wp_y):
            reward -= 100.0
            terminated = True
            info["collision"] = True

        # r_path_error (A* path deviation — 0 when no path)
        path_err = self._dist_to_nearest_astar_wp()
        reward -= 1.0 * path_err

        # r_threat_proximity (max threat score — 0 in Stages 1-2)
        max_threat = float(np.max(self.threat_scores))
        reward -= 2.0 * max_threat

        # r_smooth
        action_delta = float(np.linalg.norm(action - self.prev_action))
        reward -= 0.3 * action_delta

        # r_time
        reward -= 0.05

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

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self._running = False
        time.sleep(0.2)
        if self.node:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
