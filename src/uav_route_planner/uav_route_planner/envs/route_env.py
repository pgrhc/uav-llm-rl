#!/usr/bin/env python3
"""
Route Planning Agent — Gazebo-based Gymnasium Environment

Observation:
    costmap_patch  : (1, 64, 64)  — CNN input from /route/costmap_patch
    threat_vector  : (88,)        — MLP input from /threat/state_vec
    threat_scores  : (5,)         — MLP input from /threat/output_scores (tehdit ajanı)
    goal_state     : (7,)         — MLP input (rel_goal, dist, speed, yaw)

Action:
    Box(4,) → (dx, dy, dz, dyaw)  scaled to physical limits

Reward:
    Goal progress, collision penalty, threat proximity penalty,
    smoothness penalty, goal reached bonus, time penalty.
"""

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
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2


class RouteEnv(gym.Env):
    metadata = {"render_modes": []}

    # Physical limits
    STEP_SIZE = 0.3        # max horizontal delta (m)
    Z_STEP = 0.2           # max vertical delta (m)
    MAX_YAW_RATE = 0.52    # ~30 deg in radians
    GOAL_TOLERANCE = 0.5   # m — episode success
    SAFETY_RADIUS = 0.5    # m — min obstacle distance
    LETHAL_THRESHOLD = 90  # costmap value (0-100)
    MAX_EPISODE_STEPS = 500
    MAX_GOAL_DIST = 30.0   # for normalization
    MAX_SPEED = 5.0        # for normalization

    def __init__(self):
        super().__init__()

        # --- Action space: (dx, dy, dz, dyaw) in [-1, 1] ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # --- Observation space ---
        self.observation_space = spaces.Dict({
            "costmap_patch": spaces.Box(
                low=0.0, high=1.0, shape=(1, 64, 64), dtype=np.float32
            ),
            "threat_vector": spaces.Box(
                low=-np.inf, high=np.inf, shape=(88,), dtype=np.float32
            ),
            "threat_scores": spaces.Box(
                low=0.0, high=1.0, shape=(5,), dtype=np.float32
            ),
            "goal_state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
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
        self.threat_vector = np.zeros(88, dtype=np.float32)
        self.threat_scores = np.zeros(5, dtype=np.float32)  # /threat/output_scores
        self.latest_costmap = None

        # Freshness flags (threat_scores optional — tehdit ajanı yoksa 0 kalır)
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
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        qos_transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        # Subscribers
        self.node.create_subscription(
            Image, "/route/costmap_patch", self._cb_patch, 10
        )
        self.node.create_subscription(
            Float32MultiArray, "/threat/state_vec", self._cb_threat, 10
        )
        self.node.create_subscription(
            Float32MultiArray, "/threat/output_scores", self._cb_threat_scores, 10
        )
        self.node.create_subscription(
            Odometry, "/odometry/filtered", self._cb_odom, qos_sensor
        )
        self.node.create_subscription(
            PoseStamped, "/goal_pose", self._cb_goal, qos_transient
        )
        self.node.create_subscription(
            OccupancyGrid, "/local_costmap/costmap",
            self._cb_costmap, qos_reliable
        )

        # Publisher — RL agent's desired waypoint
        self.waypoint_pub = self.node.create_publisher(
            PoseStamped, "/route/waypoint_desired", 10
        )

        # Spin thread
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
        if data.shape[0] == 88:
            with self.cond:
                self.threat_vector = data
                self.new_threat = True
                self.cond.notify_all()

    def _cb_threat_scores(self, msg: Float32MultiArray):
        """Tehdit ajanından gelen Top-K skorlar (0–1). Ajan çalışmıyorsa 0 kalır."""
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

    def _get_obs(self) -> dict:
        return {
            "costmap_patch": self.costmap_patch.copy(),
            "threat_vector": self.threat_vector.copy(),
            "threat_scores": np.clip(self.threat_scores.copy(), 0.0, 1.0),
            "goal_state": self._build_goal_state(),
        }

    def _dist_to_goal(self) -> float:
        if self.goal_x is None:
            return float("inf")
        dx = self.goal_x - self.drone_x
        dy = self.goal_y - self.drone_y
        return math.sqrt(dx * dx + dy * dy)

    def _check_collision(self, x: float, y: float) -> bool:
        """Check if (x, y) hits a lethal cell in the raw costmap."""
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

    def _threat_proximity_penalty(self) -> float:
        """Penalty based on nearest threat distance from threat_vector."""
        vec = self.threat_vector
        penalty = 0.0
        for i in range(5):
            base = 3 + i * 17
            is_valid = vec[base + 16] if base + 16 < len(vec) else 0.0
            if is_valid < 0.5:
                continue
            r_3d = vec[base + 4]
            if r_3d < 3.0:
                penalty += (3.0 - r_3d) / 3.0
        return penalty

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

        # Scale action to physical units
        dx = float(action[0]) * self.STEP_SIZE
        dy = float(action[1]) * self.STEP_SIZE
        dz = float(action[2]) * self.Z_STEP
        dyaw = float(action[3]) * self.MAX_YAW_RATE

        # Compute absolute waypoint
        cos_yaw = math.cos(self.drone_yaw)
        sin_yaw = math.sin(self.drone_yaw)
        world_dx = cos_yaw * dx - sin_yaw * dy
        world_dy = sin_yaw * dx + cos_yaw * dy

        wp_x = self.drone_x + world_dx
        wp_y = self.drone_y + world_dy
        wp_z = self.drone_z + dz
        wp_yaw = self.drone_yaw + dyaw

        # Publish waypoint to the rest of the pipeline
        self._publish_waypoint(wp_x, wp_y, wp_z, wp_yaw)

        # Wait for sensors to update after movement
        self._wait_obs(timeout=0.3)

        # --- Reward computation ---
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        dist = self._dist_to_goal()

        # 1. Goal progress
        if self.prev_dist_to_goal is not None and self.goal_x is not None:
            progress = self.prev_dist_to_goal - dist
            reward += 2.0 * progress

        # 2. Goal reached
        if dist < self.GOAL_TOLERANCE:
            reward += 50.0
            terminated = True
            info["success"] = True

        # 3. Collision
        if self._check_collision(wp_x, wp_y):
            reward -= 100.0
            terminated = True
            info["collision"] = True

        # 4. Threat proximity
        threat_pen = self._threat_proximity_penalty()
        reward -= 2.0 * threat_pen

        # 5. Smoothness
        action_delta = np.linalg.norm(action - self.prev_action)
        reward -= 0.3 * action_delta

        # 6. Time penalty
        reward -= 0.1

        # 7. Truncation (max steps)
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
