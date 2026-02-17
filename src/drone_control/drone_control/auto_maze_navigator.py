#!/usr/bin/env python3
"""
auto_maze_navigator.py
─────────────────────
Drone'u maze içinde otomatik olarak A* ile gezdiren ROS2 node'u.

Çalışma mantığı:
  1. Maze bilgisini (walls) dışarıdan alır veya kendi üretir.
  2. Mevcut hücreyi odometry'den hesaplar.
  3. Rastgele bir hedef hücre seçer, A* ile yol bulur.
  4. Waypoint'leri /plan topic'ine publish eder.
  5. Hedefe ulaşınca yeni hedef seçer → sonsuz döngü.

Bu sayede RViz2'den manuel goal vermek gerekmez.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped

import math
import heapq
import random
import time
import json
import os


# ── Maze parametreleri (maze_with_dynamic_obstacles.py ile aynı olmalı) ──
ROWS       = 10
COLS       = 10
CELL_SIZE  = 5.0
DRONE_SPAWN_CELL = (ROWS // 2, COLS // 2)   # (5, 5)

# Dünya koordinatlarına dönüşüm için offset
SPAWN_R, SPAWN_C = DRONE_SPAWN_CELL
ORIGIN_X = -(SPAWN_C + 0.5) * CELL_SIZE   # ENU X
ORIGIN_Y = -(SPAWN_R + 0.5) * CELL_SIZE   # ENU Y

MISSION_ALT   = -5.0   # NED Z (negatif = yukarı)
ARRIVAL_DIST  = 2.5    # Hedefe bu kadar yaklaşınca "ulaşıldı" say (metre)

# Walls dosyası: maze scripti çalıştıktan sonra bu dosyaya kaydedilecek
WALLS_FILE = "/home/ubuntu/Desktop/maze_walls.json"


# ═══════════════════════════════════════════════════════════════════════════
# YARDIMCI: Hücre ↔ Dünya Koordinat Dönüşümleri
# ═══════════════════════════════════════════════════════════════════════════

def cell_to_world(r: int, c: int):
    """Hücre (row, col) → Dünya ENU koordinatı (x, y)"""
    x = ORIGIN_X + (c + 0.5) * CELL_SIZE
    y = ORIGIN_Y + (r + 0.5) * CELL_SIZE
    return x, y

def world_to_cell(x: float, y: float):
    """Dünya ENU koordinatı (x, y) → En yakın hücre (row, col)"""
    c = int((x - ORIGIN_X) / CELL_SIZE)
    r = int((y - ORIGIN_Y) / CELL_SIZE)
    r = max(0, min(ROWS - 1, r))
    c = max(0, min(COLS - 1, c))
    return r, c


# ═══════════════════════════════════════════════════════════════════════════
# A* ALGORİTMASI
# ═══════════════════════════════════════════════════════════════════════════

DIRS_RC = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

def astar(walls, start: tuple, goal: tuple):
    """
    walls: maze_with_dynamic_obstacles.py'deki 'walls' listesi
    start/goal: (row, col)
    Döndürür: [(row,col), ...] listesi (başlangıç dahil, hedef dahil)
    """
    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap = []
    heapq.heappush(open_heap, (h(start, goal), 0, start))

    came_from = {start: None}
    g_cost    = {start: 0}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == goal:
            # Yolu geri izle
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        r, c = current
        for direction, (dr, dc) in DIRS_RC.items():
            # Bu yönde duvar var mı?
            if walls[r][c][direction]:
                continue
            neighbor = (r + dr, c + dc)
            if not (0 <= neighbor[0] < ROWS and 0 <= neighbor[1] < COLS):
                continue
            new_g = g + 1
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                came_from[neighbor] = current
                f = new_g + h(neighbor, goal)
                heapq.heappush(open_heap, (f, new_g, neighbor))

    return []   # Yol bulunamadı


# ═══════════════════════════════════════════════════════════════════════════
# DUVAR YÜKLEYİCİ
# ═══════════════════════════════════════════════════════════════════════════

def load_walls_from_file(path: str):
    """maze_with_dynamic_obstacles.py'nin kaydettiği walls JSON'ını yükle."""
    with open(path, "r") as f:
        raw = json.load(f)
    # JSON key'leri string, biz aynı formatı döndürüyoruz
    return raw

def generate_fallback_maze():
    """
    Walls dosyası yoksa basit bir maze üret
    (maze_with_dynamic_obstacles.py'deki algoritmanın kopyası).
    """
    import random as _r

    walls = [[{"N": True, "E": True, "S": True, "W": True}
              for _ in range(COLS)] for _ in range(ROWS)]
    vis = [[False] * COLS for _ in range(ROWS)]
    stack = [(0, 0)]
    vis[0][0] = True

    OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}

    while stack:
        r, c = stack[-1]
        neigh = []
        for d, (dr, dc) in DIRS_RC.items():
            rr, cc = r + dr, c + dc
            if 0 <= rr < ROWS and 0 <= cc < COLS and not vis[rr][cc]:
                neigh.append((d, rr, cc))
        if not neigh:
            stack.pop()
            continue
        d, rr, cc = _r.choice(neigh)
        walls[r][c][d] = False
        walls[rr][cc][OPP[d]] = False
        vis[rr][cc] = True
        stack.append((rr, cc))

    # Giriş/çıkış aç
    walls[0][0]["N"] = False
    walls[ROWS - 1][COLS - 1]["S"] = False

    # Spawn hücresi etrafını aç
    sr, sc = DRONE_SPAWN_CELL
    for d in ["N", "S", "E", "W"]:
        walls[sr][sc][d] = False

    return walls


# ═══════════════════════════════════════════════════════════════════════════
# ANA NODE
# ═══════════════════════════════════════════════════════════════════════════

class AutoMazeNavigator(Node):

    def __init__(self):
        super().__init__("auto_maze_navigator")

        # QoS
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publisher: /plan → follow_path.py dinliyor
        self.path_pub = self.create_publisher(Path, "/plan", qos_reliable)

        # Subscriber: Odometry
        self.odom_sub = self.create_subscription(
            Odometry, "/odometry/filtered", self.odom_callback, qos_be)

        # Durum
        self.current_pos_enu = [0.0, 0.0, 0.0]
        self.odom_received    = False
        self.navigating       = False
        self.current_goal_cell = None
        self.path_cells        = []
        self.current_wp_idx    = 0

        # Maze duvarlarını yükle
        self.walls = self._load_walls()
        self.get_logger().info(
            f"Maze yüklendi: {ROWS}x{COLS}, "
            f"Spawn: {DRONE_SPAWN_CELL}"
        )

        # 2 saniyede bir durum kontrolü
        self.timer = self.create_timer(2.0, self.navigation_loop)
        self.get_logger().info("AutoMazeNavigator başlatıldı. Odometry bekleniyor...")

    # ── Duvar Yükleme ──────────────────────────────────────────────────────

    def _load_walls(self):
        if os.path.exists(WALLS_FILE):
            try:
                walls = load_walls_from_file(WALLS_FILE)
                self.get_logger().info(f"Walls dosyadan yüklendi: {WALLS_FILE}")
                return walls
            except Exception as e:
                self.get_logger().warn(f"Walls dosyası okunamadı: {e}, fallback üretiliyor.")
        else:
            self.get_logger().warn(
                f"{WALLS_FILE} bulunamadı. "
                "maze_with_dynamic_obstacles.py'ye save_walls() ekle! "
                "Şimdilik rastgele maze üretiliyor."
            )
        return generate_fallback_maze()

    # ── Odometry ───────────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry):
        self.current_pos_enu = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]
        self.odom_received = True

    # ── Ana Navigasyon Döngüsü ─────────────────────────────────────────────

    def navigation_loop(self):
        if not self.odom_received:
            self.get_logger().info("Odometry bekleniyor...")
            return

        curr_x, curr_y, _ = self.current_pos_enu
        curr_cell = world_to_cell(curr_x, curr_y)

        # Hedefe ulaşıldı mı?
        if self.navigating and self.current_goal_cell is not None:
            goal_x, goal_y = cell_to_world(*self.current_goal_cell)
            dist = math.sqrt((curr_x - goal_x)**2 + (curr_y - goal_y)**2)

            if dist < ARRIVAL_DIST:
                self.get_logger().info(
                    f"✅ Hedefe ulaşıldı! "
                    f"Hücre: {self.current_goal_cell}, "
                    f"Dist: {dist:.2f}m"
                )
                self.navigating = False
                # Kısa bekleme
                time.sleep(0.5)

        # Yeni hedef seç ve yol planla
        if not self.navigating:
            self._plan_new_goal(curr_cell)

    def _plan_new_goal(self, start_cell: tuple):
        """Rastgele bir hedef seç, A* ile yol bul, /plan'a publish et."""

        # Spawn hücresinden uzak bir hedef seç
        attempts = 0
        goal_cell = start_cell

        while goal_cell == start_cell or self._manhattan(goal_cell, start_cell) < 3:
            goal_cell = (
                random.randint(0, ROWS - 1),
                random.randint(0, COLS - 1)
            )
            attempts += 1
            if attempts > 50:
                self.get_logger().warn("Uygun hedef bulunamadı, spawn hücresi kullanılıyor.")
                goal_cell = (0, 0)
                break

        self.get_logger().info(
            f"🎯 Yeni hedef: {goal_cell} "
            f"(Start: {start_cell}, Manhattan: {self._manhattan(start_cell, goal_cell)})"
        )

        # A* ile yol bul
        path_cells = astar(self.walls, start_cell, goal_cell)

        if not path_cells:
            self.get_logger().warn(f"A* yol bulamadı: {start_cell} → {goal_cell}")
            return

        self.get_logger().info(f"🗺️  A* yol uzunluğu: {len(path_cells)} hücre")

        # /plan için Path mesajı oluştur
        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for (r, c) in path_cells:
            wx, wy = cell_to_world(r, c)
            pose = PoseStamped()
            pose.header.stamp    = path_msg.header.stamp
            pose.header.frame_id = "map"
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0   # follow_path.py kendi altitude'unu kullanır
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info(
            f"📤 /plan publish edildi: {len(path_msg.poses)} waypoint"
        )

        self.current_goal_cell = goal_cell
        self.navigating        = True

    @staticmethod
    def _manhattan(a: tuple, b: tuple) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ═══════════════════════════════════════════════════════════════════════════
# MAZE SCRIPT'E EKLENECEKLERİ (maze_with_dynamic_obstacles.py sonuna ekle)
# ═══════════════════════════════════════════════════════════════════════════
#
# import json
#
# def save_walls(walls, path="/tmp/maze_walls.json"):
#     """Walls verisini JSON'a kaydet → auto_maze_navigator okuyacak."""
#     with open(path, "w") as f:
#         json.dump(walls, f)
#     print(f"✅ Walls kaydedildi: {path}")
#
# if __name__ == "__main__":
#     walls = generate_perfect_maze(ROWS, COLS)
#     save_walls(walls)                    # ← BU SATIRI EKLE
#     segments = maze_to_segments(...)
#     ...


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

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