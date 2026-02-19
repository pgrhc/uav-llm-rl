#!/usr/bin/env python3
"""
auto_maze_navigator.py  —  Frontier-Based Exploration
──────────────────────────────────────────────────────
Eski: Rastgele hedef seç → direkt git (keşif yok)
Yeni: Hiç gidilmemiş komşu hücreleri (frontier) önceliklendir
      → Drone maze'i sistematik olarak tarar, tüm koridorları gezir.

Strateji:
  1. Ziyaret edilmemiş hücreleri takip et (visited set).
  2. Her adımda mevcut hücrenin erişilebilir komşularından
     ZİYARET EDİLMEMİŞ olanları "frontier" olarak listele.
  3. Frontier boşsa uzak bir ziyaret edilmemiş hücreye git.
  4. Her yer gezilince yeniden başla (sıfırla).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped

import math
import heapq
import random
import json
import os


# ── Maze parametreleri ────────────────────────────────────────────────────────
ROWS             = 15
COLS             = 15
CELL_SIZE        = 5.0
DRONE_SPAWN_CELL = (ROWS // 2, COLS // 2)

SPAWN_R, SPAWN_C = DRONE_SPAWN_CELL
ORIGIN_X = -(SPAWN_C + 0.5) * CELL_SIZE
ORIGIN_Y = -(SPAWN_R + 0.5) * CELL_SIZE

ARRIVAL_DIST  = 2.0    # Hedefe bu kadar yaklaşınca "ulaşıldı" say

WALLS_FILE = "/home/ubuntu/Desktop/maze_walls.json"

DIRS_RC = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


# ── Koordinat dönüşümleri ─────────────────────────────────────────────────────

def cell_to_world(r, c):
    x = ORIGIN_X + (c + 0.5) * CELL_SIZE
    y = ORIGIN_Y + (r + 0.5) * CELL_SIZE
    return x, y

def world_to_cell(x, y):
    c = int((x - ORIGIN_X) / CELL_SIZE)
    r = int((y - ORIGIN_Y) / CELL_SIZE)
    return max(0, min(ROWS-1, r)), max(0, min(COLS-1, c))


# ── A* ────────────────────────────────────────────────────────────────────────

def astar(walls, start, goal):
    def h(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

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
            nb = (r+dr, c+dc)
            if not (0 <= nb[0] < ROWS and 0 <= nb[1] < COLS):
                continue
            ng = cost + 1
            if nb not in g or ng < g[nb]:
                g[nb] = ng
                came_from[nb] = cur
                heapq.heappush(heap, (ng + h(nb, goal), ng, nb))
    return []


# ── Walls yükleme ─────────────────────────────────────────────────────────────

def load_walls():
    if os.path.exists(WALLS_FILE):
        try:
            with open(WALLS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback: basit maze üret
    walls = [[{"N":True,"E":True,"S":True,"W":True} for _ in range(COLS)]
             for _ in range(ROWS)]
    vis   = [[False]*COLS for _ in range(ROWS)]
    OPP   = {"N":"S","S":"N","E":"W","W":"E"}
    stack = [(0,0)]
    vis[0][0] = True
    while stack:
        r,c = stack[-1]
        nb = [(d,r+dr,c+dc) for d,(dr,dc) in DIRS_RC.items()
              if 0<=r+dr<ROWS and 0<=c+dc<COLS and not vis[r+dr][c+dc]]
        if not nb:
            stack.pop(); continue
        d,rr,cc = random.choice(nb)
        walls[r][c][d] = False
        walls[rr][cc][OPP[d]] = False
        vis[rr][cc] = True
        stack.append((rr,cc))
    walls[0][0]["N"] = False
    walls[ROWS-1][COLS-1]["S"] = False
    sr,sc = DRONE_SPAWN_CELL
    for d in ["N","S","E","W"]:
        walls[sr][sc][d] = False
    return walls


# ── Ana Node ──────────────────────────────────────────────────────────────────

class AutoMazeNavigator(Node):

    def __init__(self):
        super().__init__("auto_maze_navigator")

        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        self.path_pub = self.create_publisher(Path, "/plan", qos_rel)
        self.odom_sub = self.create_subscription(
            Odometry, "/odometry/filtered", self.odom_cb, qos_be)

        self.pos_enu    = [0.0, 0.0, 0.0]
        self.odom_ok    = False
        self.navigating = False
        self.goal_cell  = None

        self.walls   = load_walls()
        self.visited = set()
        self.visited.add(DRONE_SPAWN_CELL)

        self.get_logger().info(
            f"Maze yüklendi {ROWS}x{COLS}. "
            f"Toplam {ROWS*COLS} hücre keşfedilecek."
        )

        self.timer = self.create_timer(1.5, self.loop)

    # ── Odometry ──────────────────────────────────────────────────────────────

    def odom_cb(self, msg):
        self.pos_enu = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]
        self.odom_ok = True
        self.visited.add(world_to_cell(self.pos_enu[0], self.pos_enu[1]))

    # ── Ana Döngü ─────────────────────────────────────────────────────────────

    def loop(self):
        if not self.odom_ok:
            return

        curr_cell = world_to_cell(self.pos_enu[0], self.pos_enu[1])

        if self.navigating and self.goal_cell:
            gx, gy = cell_to_world(*self.goal_cell)
            dist = math.hypot(self.pos_enu[0]-gx, self.pos_enu[1]-gy)
            if dist < ARRIVAL_DIST:
                self.get_logger().info(
                    f"✅ Ulaşıldı: {self.goal_cell}  "
                    f"Gezilen: {len(self.visited)}/{ROWS*COLS} hücre"
                )
                self.visited.add(self.goal_cell)
                self.navigating = False

        if not self.navigating:
            self._pick_next_goal(curr_cell)

    # ── Hedef Seçimi (Frontier-Based) ─────────────────────────────────────────

    def _pick_next_goal(self, curr_cell):
        """
        Öncelik sırası:
          1. Mevcut hücreden duvarsız geçişle ulaşılabilen ziyaret edilmemiş komşu
          2. Tüm maze'den en yakın ziyaret edilmemiş hücre (A* ile gidilir)
          3. Hepsi gezilmişse sıfırla
        """
        frontiers = self._get_frontiers(curr_cell)

        if frontiers:
            goal   = random.choice(frontiers)
            reason = "komşu frontier"
        else:
            goal = self._nearest_unvisited(curr_cell)
            if goal:
                reason = "uzak frontier"
            else:
                self.get_logger().info("🏁 Tüm maze gezildi! Keşif sıfırlanıyor...")
                self.visited.clear()
                self.visited.add(curr_cell)
                goal   = self._nearest_unvisited(curr_cell)
                reason = "yeniden başlangıç"

        if goal is None:
            return

        path_cells = astar(self.walls, curr_cell, goal)
        if not path_cells:
            self.get_logger().warn(f"A* yol bulamadı: {curr_cell} → {goal}")
            self.visited.add(goal)
            return

        self.get_logger().info(
            f"🎯 Hedef: {goal} ({reason})  "
            f"Yol: {len(path_cells)} adım  "
            f"Gezilen: {len(self.visited)}/{ROWS*COLS}"
        )

        self._publish_path(path_cells)
        self.goal_cell  = goal
        self.navigating = True

    def _get_frontiers(self, cell):
        """Duvarsız geçişle ulaşılabilen, ziyaret edilmemiş komşular."""
        r, c = cell
        return [
            (r+dr, c+dc)
            for d, (dr, dc) in DIRS_RC.items()
            if not self.walls[r][c][d]
            and 0 <= r+dr < ROWS and 0 <= c+dc < COLS
            and (r+dr, c+dc) not in self.visited
        ]

    def _nearest_unvisited(self, curr_cell):
        """Tüm ziyaret edilmemişler arasından Manhattan mesafesiyle en yakın 3'ten biri."""
        all_cells = {(r, c) for r in range(ROWS) for c in range(COLS)}
        unvisited = sorted(
            all_cells - self.visited,
            key=lambda nb: abs(nb[0]-curr_cell[0]) + abs(nb[1]-curr_cell[1])
        )
        if not unvisited:
            return None
        candidates = unvisited[:min(3, len(unvisited))]
        return random.choice(candidates)

    # ── Path Publish ──────────────────────────────────────────────────────────

    def _publish_path(self, path_cells):
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        for (r, c) in path_cells:
            wx, wy = cell_to_world(r, c)
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0   # follow_path.py kendi altitude'unu kullanır
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = AutoMazeNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()