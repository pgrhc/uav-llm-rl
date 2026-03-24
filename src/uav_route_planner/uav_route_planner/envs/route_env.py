#!/usr/bin/env python3
"""
Route Planning Agent — Gazebo-based Gymnasium Environment  (Faz 1)

Observation:
    costmap_patch  : (1, 64, 64)  — CNN /route/costmap_patch (costmap_patch_node: varsayılan harita eksenli)
    threat_vector  : (74,)        — MLP input   /threat/state_vec
    threat_scores  : (5,)         — MLP input   /threat/target_scores
    goal_state     : (7,)         — MLP input   (rel_goal, dist, speed, yaw)
    a_star_path    : (10,)        — MLP input   /plan  (5 wp × rel_x,rel_y)

Action:
    Box(4,) → (dx, dy, dz, dyaw)  scaled to physical limits (ROUTE_STEP_SIZE, varsayılan 0.3 m)

Reward  (Faz 1 — Çekirdek):
    r_progress   +2.0 × (prev_dist − curr_dist)
    r_goal       +50.0
    r_collision  −100.0   (costmap lethal VEYA LiDAR < 0.4 m)
    r_time       −0.1
    r_smooth     −0.3 × ‖Δaction‖
"""

import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2


class RouteEnv(gym.Env):
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

    def __init__(self):
        super().__init__()

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

        # --- Internal state ---
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

        # Sensor buffers
        self.costmap_patch = np.zeros((1, 64, 64), dtype=np.float32)
        self.threat_vector = np.zeros(74, dtype=np.float32)
        self.threat_scores = np.zeros(5, dtype=np.float32)
        self.a_star_poses = []
        self.latest_costmap = None

        # Freshness flags
        self.new_patch = False
        self.new_threat = False
        self.new_odom = False
        self.cond = threading.Condition()

        self.bridge = CvBridge()

        # --- ROS 2 ---
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("route_env_node")
        self._running = True

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        qos_transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        qos_costmap = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10,
        )

        self.node.create_subscription(
            Image, "/route/costmap_patch", self._cb_patch, 10
        )
        self.node.create_subscription(
            Float32MultiArray, "/threat/state_vec", self._cb_threat, 10
        )
        self.node.create_subscription(
            Float32MultiArray, "/threat/target_scores", self._cb_threat_scores, 10
        )
        self.node.create_subscription(
            Odometry, "/odometry/filtered", self._cb_odom, qos_sensor
        )
        self.node.create_subscription(
            PoseStamped, "/goal_pose", self._cb_goal, qos_transient
        )
        self.node.create_subscription(
            OccupancyGrid, "/local_costmap/costmap",
            self._cb_costmap, qos_costmap
        )
        self.node.create_subscription(
            Path, "/plan", self._cb_plan, qos_sensor
        )

        self.waypoint_pub = self.node.create_publisher(
            PoseStamped, "/route/waypoint_desired", 10
        )

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        time.sleep(1.0)

    # ------------------------------------------------------------------ #
    # ROS spin
    # ------------------------------------------------------------------ #
    def _spin(self):
        while self._running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
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
        self.node.get_logger().info(
            f"Goal received: ({self.goal_x:.1f}, {self.goal_y:.1f}, {self.goal_z:.1f})"
        )

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
            dist_norm,
            speed_norm,
            math.sin(self.drone_yaw),
            math.cos(self.drone_yaw),
        ], dtype=np.float32)

    def _build_a_star_obs(self) -> np.ndarray:
        """Drone'un önündeki en yakın 5 A* waypoint'ini relatif koordinat olarak döndür."""
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

    def _check_collision(self, wp_x: float, wp_y: float) -> bool:
        """Costmap lethal (waypoint veya drone) VEYA LiDAR < 0.4 m."""
        with self.cond:
            dx, dy = self.drone_x, self.drone_y
        if self._check_costmap_collision(wp_x, wp_y) or self._check_costmap_collision(
            dx, dy
        ):
            return True
        if self._check_lidar_collision():
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

        # ---------- Faz 1 reward ----------
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

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

        # r_smooth
        action_delta = np.linalg.norm(action - self.prev_action)
        reward -= 0.3 * action_delta

        # r_time
        reward -= 0.05

        # truncation
        if self.step_count >= self.MAX_EPISODE_STEPS:
            truncated = True
            info["timeout"] = True

        self.prev_dist_to_goal = dist
        self.prev_action = action.copy()

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self._running = False
        time.sleep(0.2)
        if self.node:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
