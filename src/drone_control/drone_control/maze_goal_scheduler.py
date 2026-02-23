#!/usr/bin/env python3
"""
maze_goal_scheduler.py — Sadece /goal_pose yayınlayan hedef planlayıcı

Rota ajanı eğitimi için: Drone'u hareket ettirmez, sadece sıradaki hedefi
/goal_pose topic'ine yazar. Drone'u rota ajanı (veya başka bir controller)
sürer; bu node sadece frontier mantığıyla bir sonraki hedef hücreyi seçer.

Kullanım: Rota ajanı eğitirken bu node'u çalıştır → sürekli yeni hedefler
gelir, rota ajanı o hedeflere gitmeye çalışır.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

import math
import random
import json
import os


# ── Maze parametreleri (auto_maze_navigator ile aynı) ─────────────────────────
ROWS             = 15
COLS             = 15
CELL_SIZE        = 5.0
DRONE_SPAWN_CELL = (ROWS // 2, COLS // 2)

SPAWN_R, SPAWN_C = DRONE_SPAWN_CELL
ORIGIN_X = -(SPAWN_C + 0.5) * CELL_SIZE
ORIGIN_Y = -(SPAWN_R + 0.5) * CELL_SIZE

ARRIVAL_DIST = 2.0   # Hedefe bu kadar yaklaşınca "ulaşıldı" say
GOAL_Z       = 1.5   # Hedef yüksekliği (metre)

WALLS_FILE = "/home/ubuntu/Desktop/maze_walls.json"

DIRS_RC = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


def cell_to_world(r, c):
    x = ORIGIN_X + (c + 0.5) * CELL_SIZE
    y = ORIGIN_Y + (r + 0.5) * CELL_SIZE
    return x, y


def world_to_cell(x, y):
    c = int((x - ORIGIN_X) / CELL_SIZE)
    r = int((y - ORIGIN_Y) / CELL_SIZE)
    return max(0, min(ROWS - 1, r)), max(0, min(COLS - 1, c))


def load_walls():
    if os.path.exists(WALLS_FILE):
        try:
            with open(WALLS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback: basit maze
    walls = [[{"N": True, "E": True, "S": True, "W": True} for _ in range(COLS)]
             for _ in range(ROWS)]
    vis = [[False] * COLS for _ in range(ROWS)]
    OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
    stack = [(0, 0)]
    vis[0][0] = True
    while stack:
        r, c = stack[-1]
        nb = [(d, r + dr, c + dc) for d, (dr, dc) in DIRS_RC.items()
              if 0 <= r + dr < ROWS and 0 <= c + dc < COLS and not vis[r + dr][c + dc]]
        if not nb:
            stack.pop()
            continue
        d, rr, cc = random.choice(nb)
        walls[r][c][d] = False
        walls[rr][cc][OPP[d]] = False
        vis[rr][cc] = True
        stack.append((rr, cc))
    walls[0][0]["N"] = False
    walls[ROWS - 1][COLS - 1]["S"] = False
    sr, sc = DRONE_SPAWN_CELL
    for d in ["N", "S", "E", "W"]:
        walls[sr][sc][d] = False
    return walls


# ── Node ────────────────────────────────────────────────────────────────────

class MazeGoalScheduler(Node):

    def __init__(self):
        super().__init__("maze_goal_scheduler")

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_goal = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos_goal)
        self.odom_sub = self.create_subscription(
            Odometry, "/odometry/filtered", self.odom_cb, qos_be
        )

        self.pos = [0.0, 0.0, 0.0]
        self.odom_ok = False
        self.current_goal_cell = None

        self.walls = load_walls()
        self.visited = set()
        self.visited.add(DRONE_SPAWN_CELL)

        self.get_logger().info(
            f"Maze goal scheduler başladı. {ROWS}x{COLS} hücre. "
            "Sadece /goal_pose yayınlıyor (rota ajanı eğitimi için)."
        )

        self.timer = self.create_timer(1.5, self.loop)

    def odom_cb(self, msg):
        self.pos[0] = msg.pose.pose.position.x
        self.pos[1] = msg.pose.pose.position.y
        self.pos[2] = msg.pose.pose.position.z
        self.odom_ok = True
        self.visited.add(world_to_cell(self.pos[0], self.pos[1]))

    def _get_frontiers(self, cell):
        r, c = cell
        return [
            (r + dr, c + dc)
            for d, (dr, dc) in DIRS_RC.items()
            if not self.walls[r][c][d]
            and 0 <= r + dr < ROWS and 0 <= c + dc < COLS
            and (r + dr, c + dc) not in self.visited
        ]

    def _nearest_unvisited(self, curr_cell):
        all_cells = {(r, c) for r in range(ROWS) for c in range(COLS)}
        unvisited = sorted(
            all_cells - self.visited,
            key=lambda nb: abs(nb[0] - curr_cell[0]) + abs(nb[1] - curr_cell[1]),
        )
        if not unvisited:
            return None
        candidates = unvisited[: min(3, len(unvisited))]
        return random.choice(candidates)

    def _publish_goal(self, r, c):
        x, y = cell_to_world(r, c)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = GOAL_Z
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)
        self.get_logger().info(
            f"🎯 Yeni hedef: hücre ({r},{c}) = ({x:.1f}, {y:.1f})  "
            f"Gezilen: {len(self.visited)}/{ROWS*COLS}"
        )

    def loop(self):
        if not self.odom_ok:
            return

        curr_cell = world_to_cell(self.pos[0], self.pos[1])

        # Mevcut hedefe ulaşıldı mı?
        if self.current_goal_cell:
            gx, gy = cell_to_world(*self.current_goal_cell)
            dist = math.hypot(self.pos[0] - gx, self.pos[1] - gy)
            if dist < ARRIVAL_DIST:
                self.get_logger().info(
                    f"✅ Hedefe ulaşıldı: {self.current_goal_cell}"
                )
                self.visited.add(self.current_goal_cell)
                self.current_goal_cell = None

        # Yeni hedef seç ve yayınla
        if self.current_goal_cell is None:
            frontiers = self._get_frontiers(curr_cell)
            if frontiers:
                goal = random.choice(frontiers)
            else:
                goal = self._nearest_unvisited(curr_cell)
                if goal is None:
                    self.get_logger().info("🏁 Tüm maze gezildi. Keşif sıfırlanıyor...")
                    self.visited.clear()
                    self.visited.add(curr_cell)
                    goal = self._nearest_unvisited(curr_cell)
            if goal is not None:
                self.current_goal_cell = goal
                self._publish_goal(*goal)


def main(args=None):
    rclpy.init(args=args)
    node = MazeGoalScheduler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
