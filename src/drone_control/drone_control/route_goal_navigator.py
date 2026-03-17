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

# Curriculum maze sabitleri (maze_curriculum_world ile uyumlu)
ROWS, COLS = 15, 15
CELL_SIZE = 5.0
DRONE_SPAWN_CELL = (ROWS // 2, COLS // 2)
WALLS_DIR = "/home/ubuntu/Desktop"
STAGE_ORIGINS = {
    1: (0.0, 0.0),
    2: (1000.0, 0.0),
    3: (2000.0, 0.0),
}
WALLS_PATHS = {
    1: os.path.join(WALLS_DIR, "maze_walls_stage1.json"),
    2: os.path.join(WALLS_DIR, "maze_walls_stage2.json"),
    3: os.path.join(WALLS_DIR, "maze_walls_stage3.json"),
}

# Birlesik maze
UNIFIED_MAZE = True
UNIFIED_ROWS, UNIFIED_COLS = 15, 45
UNIFIED_ORIGIN = (0.0, 0.0)
UNIFIED_SPAWN_CELL = (7, 7)
WALLS_UNIFIED_PATH = os.path.join(WALLS_DIR, "maze_walls_unified.json")
SECTION1_X_MAX = 50.0
SECTION2_X_MAX = 125.0


def _maze_origin(stage_origin):
    sr, sc = DRONE_SPAWN_CELL
    ox = stage_origin[0] - (sc + 0.5) * CELL_SIZE
    oy = stage_origin[1] - (sr + 0.5) * CELL_SIZE
    return ox, oy


def _maze_origin_unified():
    sr, sc = UNIFIED_SPAWN_CELL
    ox = UNIFIED_ORIGIN[0] - (sc + 0.5) * CELL_SIZE
    oy = UNIFIED_ORIGIN[1] - (sr + 0.5) * CELL_SIZE
    return ox, oy


def _load_walls(path):
    with open(path) as f:
        return json.load(f)


def _stage_from_x(x):
    if x < SECTION1_X_MAX:
        return 1
    if x < SECTION2_X_MAX:
        return 2
    return 3


ARRIVAL_DIST = 2.0
GOAL_Z = 1.5
DIRS_RC = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


def _cell_to_world(r, c, stage_origin):
    ox, oy = _maze_origin(stage_origin)
    x = ox + (c + 0.5) * CELL_SIZE
    y = oy + (r + 0.5) * CELL_SIZE
    return x, y


def _cell_to_world_unified(r, c):
    ox, oy = _maze_origin_unified()
    x = ox + (c + 0.5) * CELL_SIZE
    y = oy + (r + 0.5) * CELL_SIZE
    return x, y


def _world_to_cell(x, y, stage_origin):
    ox, oy = _maze_origin(stage_origin)
    c = int((x - ox) / CELL_SIZE)
    r = int((y - oy) / CELL_SIZE)
    return max(0, min(ROWS - 1, r)), max(0, min(COLS - 1, c))


def _world_to_cell_unified(x, y):
    ox, oy = _maze_origin_unified()
    c = int((x - ox) / CELL_SIZE)
    r = int((y - oy) / CELL_SIZE)
    return max(0, min(UNIFIED_ROWS - 1, r)), max(0, min(UNIFIED_COLS - 1, c))


def astar(walls, start, goal, rows, cols):
    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    heap = [(h(start, goal), 0, start)]
    came_from = {start: None}
    g = {start: 0}

    while heap:
        _, cost, cur = heapq.heappop(heap)
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
                heapq.heappush(heap, (ng + h(nb, goal), ng, nb))
    return []


class RouteGoalNavigator(Node):

    def __init__(self):
        super().__init__("route_goal_navigator")

        self.unified = UNIFIED_MAZE
        self.current_stage = 1
        self._stage_origin = UNIFIED_ORIGIN if self.unified else STAGE_ORIGINS[self.current_stage]
        self.walls = self._load_walls_impl()
        self._rows = UNIFIED_ROWS if self.unified else ROWS
        self._cols = UNIFIED_COLS if self.unified else COLS
        self._spawn_cell = UNIFIED_SPAWN_CELL if self.unified else DRONE_SPAWN_CELL

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
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )

        self.path_pub = self.create_publisher(Path, "/plan", qos_path)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos_goal)

        self.odom_sub = self.create_subscription(
            Odometry, "/odometry/filtered", self._cb_odom, qos_odom)
        self.stage_sub = self.create_subscription(
            Int32, "/route/set_stage", self._cb_set_stage, 10)

        self.pos = [0.0, 0.0, 0.0]
        self.odom_ok = False
        self.navigating = False
        self.goal_cell = None
        self.current_path_cells = []

        self.visited = set()
        self.visited.add(self._spawn_cell)

        self.get_logger().info(
            f"RouteGoalNavigator baslatildi | Unified={self.unified} | "
            f"Stage {self.current_stage} | Origin {self._stage_origin}"
        )

        self.timer = self.create_timer(0.5, self._loop)

    # ------------------------------------------------------------------ #
    # Stage management
    # ------------------------------------------------------------------ #
    def _load_walls_impl(self):
        if self.unified:
            path = WALLS_UNIFIED_PATH
        else:
            path = WALLS_PATHS.get(self.current_stage)
        if path and os.path.exists(path):
            return _load_walls(path)
        self.get_logger().warn(f"walls.json bulunamadi: {path}")
        cols = UNIFIED_COLS if self.unified else COLS
        rows = UNIFIED_ROWS if self.unified else ROWS
        return [[{"N": True, "E": True, "S": True, "W": True}
                 for _ in range(cols)] for _ in range(rows)]

    def _cb_set_stage(self, msg: Int32):
        if self.unified:
            return  # Birlesik maze: stage pozisyondan
        new_stage = msg.data
        if new_stage == self.current_stage:
            return
        if new_stage not in STAGE_ORIGINS:
            self.get_logger().warn(f"Gecersiz stage: {new_stage}")
            return

        self.current_stage = new_stage
        self._stage_origin = STAGE_ORIGINS[new_stage]
        self.walls = self._load_walls_impl()

        self.visited.clear()
        self.visited.add(self._spawn_cell)
        self.navigating = False
        self.goal_cell = None
        self.current_path_cells = []

        self.get_logger().info(
            f"Stage degisti → {new_stage} | Origin {self._stage_origin}"
        )

    # ------------------------------------------------------------------ #
    # Odometry
    # ------------------------------------------------------------------ #
    def _cb_odom(self, msg: Odometry):
        self.pos[0] = msg.pose.pose.position.x
        self.pos[1] = msg.pose.pose.position.y
        self.pos[2] = msg.pose.pose.position.z
        self.odom_ok = True
        if self.unified:
            self.current_stage = _stage_from_x(self.pos[0])
            cell = _world_to_cell_unified(self.pos[0], self.pos[1])
        else:
            cell = _world_to_cell(self.pos[0], self.pos[1], self._stage_origin)
        self.visited.add(cell)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def _loop(self):
        if not self.odom_ok:
            return

        if self.unified:
            curr_cell = _world_to_cell_unified(self.pos[0], self.pos[1])
        else:
            curr_cell = _world_to_cell(self.pos[0], self.pos[1], self._stage_origin)

        if self.navigating and self.goal_cell:
            if self.unified:
                gx, gy = _cell_to_world_unified(*self.goal_cell)
            else:
                gx, gy = _cell_to_world(*self.goal_cell, self._stage_origin)
            dist = math.hypot(self.pos[0] - gx, self.pos[1] - gy)
            if dist < ARRIVAL_DIST:
                self.get_logger().info(
                    f"Hedefe ulasildi: {self.goal_cell} | "
                    f"Gezilen: {len(self.visited)}/{ROWS * COLS}"
                )
                self.visited.add(self.goal_cell)
                self.navigating = False
                self._pick_next_goal(curr_cell)

        if not self.navigating:
            self._pick_next_goal(curr_cell)

        if self.current_path_cells:
            self._publish_path(self.current_path_cells)

    # ------------------------------------------------------------------ #
    # Goal selection (frontier-based)
    # ------------------------------------------------------------------ #
    def _pick_next_goal(self, curr_cell):
        frontiers = self._get_frontiers(curr_cell)
        if frontiers:
            goal = random.choice(frontiers)
        else:
            goal = self._nearest_unvisited(curr_cell)
            if goal is None:
                self.visited.clear()
                self.visited.add(curr_cell)
                goal = self._nearest_unvisited(curr_cell)

        if goal is None:
            return

        path_cells = astar(self.walls, curr_cell, goal, self._rows, self._cols)
        if not path_cells:
            self.get_logger().warn(f"A* yol bulamadi: {curr_cell} -> {goal}")
            self.visited.add(goal)
            return

        self.current_path_cells = path_cells
        self._publish_path(path_cells)
        self._publish_goal(goal)
        self.goal_cell = goal
        self.navigating = True

        self.get_logger().info(
            f"Yeni hedef: {goal} | Yol: {len(path_cells)} adim | "
            f"Stage {self.current_stage}"
        )

    def _get_frontiers(self, cell):
        r, c = cell
        return [
            (r + dr, c + dc)
            for d, (dr, dc) in DIRS_RC.items()
            if not self.walls[r][c][d]
            and 0 <= r + dr < self._rows and 0 <= c + dc < self._cols
            and (r + dr, c + dc) not in self.visited
        ]

    def _nearest_unvisited(self, curr_cell):
        all_cells = {(r, c) for r in range(self._rows) for c in range(self._cols)}
        unvisited = sorted(
            all_cells - self.visited,
            key=lambda nb: abs(nb[0] - curr_cell[0]) + abs(nb[1] - curr_cell[1]),
        )
        if not unvisited:
            return None
        candidates = unvisited[:min(3, len(unvisited))]
        return random.choice(candidates)

    # ------------------------------------------------------------------ #
    # Publishing
    # ------------------------------------------------------------------ #
    def _publish_path(self, path_cells):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        for r, c in path_cells:
            if self.unified:
                wx, wy = _cell_to_world_unified(r, c)
            else:
                wx, wy = _cell_to_world(r, c, self._stage_origin)
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
        if self.unified:
            x, y = _cell_to_world_unified(r, c)
        else:
            x, y = _cell_to_world(r, c, self._stage_origin)
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
