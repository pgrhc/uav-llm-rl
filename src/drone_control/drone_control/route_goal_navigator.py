#!/usr/bin/env python3
"""
route_goal_navigator.py — Stage-aware goal & A* path publisher

Iki mod:
  - Ayri mazeler: /route/set_stage ile stage degisir, her stage farkli walls
  - Birlesik maze: Tek walls, pozisyon bazli stage, fiziksel gecis
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32

import math
import heapq
import random
import json
import os
import threading
from collections import deque

# Curriculum maze sabitleri (maze_curriculum_manager ile uyumlu)
ROWS, COLS = 15, 15
CELL_SIZE = 5.0
DRONE_SPAWN_CELL = (ROWS // 2, COLS // 2)
WALLS_PATH = "/home/ubuntu/Desktop/maze_walls.json"
ORIGIN = (0.0, 0.0)

# Minimum Manhattan-cell distance for goal selection.
# 8 cells × 5 m = 40 m minimum — forces the agent to navigate long sections.
MIN_GOAL_DIST_CELLS = 8
# How many recently visited goals to remember (avoids exact repetition).
RECENT_GOAL_MEMORY = 12

def _maze_origin():
    sr, sc = DRONE_SPAWN_CELL
    ox = ORIGIN[0] - (sc + 0.5) * CELL_SIZE
    oy = ORIGIN[1] - (sr + 0.5) * CELL_SIZE
    return ox, oy


def _load_walls():
    if not os.path.exists(WALLS_PATH):
        return [[{"N": True, "E": True, "S": True, "W": True} for _ in range(COLS)] for _ in range(ROWS)]
    with open(WALLS_PATH) as f:
        return json.load(f)

ARRIVAL_DIST = 1.5
GOAL_Z = 1.5
DIRS_RC = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

def _cell_to_world(r, c):
    ox, oy = _maze_origin()
    x = ox + (c + 0.5) * CELL_SIZE
    y = oy + (r + 0.5) * CELL_SIZE
    return x, y

def _world_to_cell(x, y):
    ox, oy = _maze_origin()
    c = int((x - ox) / CELL_SIZE)
    r = int((y - oy) / CELL_SIZE)
    return max(0, min(ROWS - 1, r)), max(0, min(COLS - 1, c))


def astar(walls, start, goal, rows, cols):
    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    step = 0
    heap = [(h(start, goal), step, 0, start)]
    came_from = {start: None}
    g = {start: 0}

    while heap:
        _, _, cost, cur = heapq.heappop(heap)
        if cur == goal:
            path, node = [], goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            return list(reversed(path))
        r, c = cur
        for d, (dr, dc) in DIRS_RC.items():
            if walls[r][c][d]:
                continue
            nb = (r + dr, c + dc)
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols):
                continue
            ng = cost + 1
            if nb not in g or ng < g[nb]:
                g[nb] = ng
                came_from[nb] = cur
                step += 1
                heapq.heappush(heap, (ng + h(nb, goal), step, ng, nb))
    return []


class RouteGoalNavigator(Node):

    def __init__(self):
        super().__init__("route_goal_navigator")

        self.current_stage = 1
        self.walls = _load_walls()
        self._rows = ROWS
        self._cols = COLS
        self._spawn_cell = DRONE_SPAWN_CELL

        qos_path = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10,
        )
        qos_odom = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10,
        )
        qos_goal = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )

        self.path_pub = self.create_publisher(Path, "/plan", qos_path)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos_goal)

        self.odom_sub = self.create_subscription(
            Odometry, "/odometry/filtered", self._cb_odom, qos_odom)
        self.stage_sub = self.create_subscription(
            Int32, "/route/set_stage", self._cb_set_stage, 10)

        self.pos = [0.0, 0.0, 0.0]
        self._pos_lock = threading.Lock()
        self.odom_ok = False
        self.navigating = False
        self.goal_cell = None
        self.current_path_cells = []

        # Far-goal memory: remember recent goals to avoid exact repetition
        self._recent_goals: deque = deque(maxlen=RECENT_GOAL_MEMORY)

        self.get_logger().info(
            f"RouteGoalNavigator baslatildi | Stage {self.current_stage} | "
            f"MIN_GOAL_DIST={MIN_GOAL_DIST_CELLS} cells ({MIN_GOAL_DIST_CELLS * CELL_SIZE:.0f}m)"
        )

        self.timer = self.create_timer(0.5, self._loop)

    # ------------------------------------------------------------------ #
    # Stage management
    # ------------------------------------------------------------------ #
    def _cb_set_stage(self, msg: Int32):
        new_stage = msg.data
        if new_stage == self.current_stage:
            return

        self.current_stage = new_stage
        self.walls = _load_walls()

        # Reset navigation state but keep goal memory for variety
        self._recent_goals.clear()
        self.navigating = False
        self.goal_cell = None
        self.current_path_cells = []

        self.get_logger().info(f"Stage degisti → {new_stage}")

    # ------------------------------------------------------------------ #
    # Odometry
    # ------------------------------------------------------------------ #
    def _cb_odom(self, msg: Odometry):
        with self._pos_lock:
            self.pos[0] = msg.pose.pose.position.x
            self.pos[1] = msg.pose.pose.position.y
            self.pos[2] = msg.pose.pose.position.z
        self.odom_ok = True

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def _loop(self):
        if not self.odom_ok:
            return

        with self._pos_lock:
            x, y = self.pos[0], self.pos[1]
        curr_cell = _world_to_cell(x, y)

        goal_cell = self.goal_cell
        if self.navigating and goal_cell:
            gx, gy = _cell_to_world(*goal_cell)
            dist = math.hypot(x - gx, y - gy)
            if dist < ARRIVAL_DIST:
                self.get_logger().info(
                    f"Hedefe ulasildi: {goal_cell}  recent_count={len(self._recent_goals)}"
                )
                self.navigating = False
                self._pick_next_goal(curr_cell)
        elif not self.navigating:
            self._pick_next_goal(curr_cell)

        if self.current_path_cells:
            self._publish_path(self.current_path_cells)


    # ------------------------------------------------------------------ #
    # Goal selection (far-goal strategy)
    # ------------------------------------------------------------------ #
    def _pick_next_goal(self, curr_cell):
        """Pick a goal cell at least MIN_GOAL_DIST_CELLS away from curr_cell.

        Strategy:
          1. Build a candidate list of ALL cells >= MIN_GOAL_DIST_CELLS away.
          2. Exclude recently used goals to prevent repetition.
          3. If no candidates remain (very small maze), halve the requirement.
          4. Validate with A*; retry up to MAX_ATTEMPTS times.
        """
        MAX_ATTEMPTS = 8
        min_dist = MIN_GOAL_DIST_CELLS

        for attempt in range(MAX_ATTEMPTS):
            candidates = [
                (r, c)
                for r in range(self._rows)
                for c in range(self._cols)
                if abs(r - curr_cell[0]) + abs(c - curr_cell[1]) >= min_dist
                and (r, c) not in self._recent_goals
            ]
            if not candidates:
                # Relax: forget recent goals or lower distance floor
                if self._recent_goals:
                    self._recent_goals.clear()
                    continue
                min_dist = max(4, min_dist // 2)
                continue

            goal = random.choice(candidates)
            path_cells = astar(self.walls, curr_cell, goal, self._rows, self._cols)
            if not path_cells:
                self.get_logger().warn(f"A* yol bulamadi: {curr_cell} → {goal}, yeniden deneniyor")
                continue  # try another random far cell

            # Commit to this goal
            self._recent_goals.append(goal)
            self.current_path_cells = path_cells
            self._publish_path(path_cells)
            self._publish_goal(goal)
            self.goal_cell = goal
            self.navigating = True

            gx, gy = _cell_to_world(*goal)
            dist_m = math.hypot(
                gx - _cell_to_world(*curr_cell)[0],
                gy - _cell_to_world(*curr_cell)[1]
            )
            self.get_logger().info(
                f"Yeni hedef: {goal}  dist={dist_m:.0f}m  yol={len(path_cells)} adim  "
                f"Stage {self.current_stage}  recent={len(self._recent_goals)}"
            )
            return

        self.get_logger().warn("Uygun uzak hedef bulunamadi, yakın fallback.")
        # Fallback: any reachable cell
        for r in range(self._rows):
            for c in range(self._cols):
                if (r, c) == curr_cell:
                    continue
                path_cells = astar(self.walls, curr_cell, (r, c), self._rows, self._cols)
                if path_cells:
                    self.current_path_cells = path_cells
                    self._publish_path(path_cells)
                    self._publish_goal((r, c))
                    self.goal_cell = (r, c)
                    self.navigating = True
                    return

    # ------------------------------------------------------------------ #
    # Publishing
    # ------------------------------------------------------------------ #
    def _publish_path(self, path_cells):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        for r, c in path_cells:
            wx, wy = _cell_to_world(r, c)
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)

    def _publish_goal(self, goal_cell):
        r, c = goal_cell
        x, y = _cell_to_world(r, c)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = GOAL_Z
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RouteGoalNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
