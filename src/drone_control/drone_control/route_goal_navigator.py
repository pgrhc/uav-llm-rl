#!/usr/bin/env python3
import json
import math
import os
import random
import threading

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry,Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, Bool


ROWS, COLS = 15, 15
CELL_SIZE = 5.0
DRONE_SPAWN_CELL = (ROWS // 2, COLS // 2)
WALLS_PATH = "/home/ubuntu/Desktop/maze_walls.json"
ORIGIN = (0.0, 0.0)
ARRIVAL_DIST = 1.0
GOAL_Z = 1.5
MIN_GOAL_DIST_CELLS = 1
MAX_GOAL_DIST_CELLS = 1
GOAL_DISTANCE_GROWTH_EPISODES = 40
INITIAL_MAX_GOAL_DIST_CELLS = 1
RECENT_GOAL_MEMORY = 10


def _maze_origin():
    sr, sc = DRONE_SPAWN_CELL
    ox = ORIGIN[0] - (sc + 0.5) * CELL_SIZE
    oy = ORIGIN[1] - (sr + 0.5) * CELL_SIZE
    return ox, oy


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


def _load_walls():
    if not os.path.exists(WALLS_PATH):
        return None
    try:
        with open(WALLS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class RouteGoalNavigator(Node):
    def __init__(self):
        super().__init__("route_goal_navigator")
        self.current_stage = 1
        self.walls = _load_walls()

        qos_odom = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_goal = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos_goal)
        self.odom_sub = self.create_subscription(Odometry, "/odometry/filtered", self._cb_odom, qos_odom)
        self.stage_sub = self.create_subscription(Int32, "/route/set_stage", self._cb_set_stage, 10)

        self.pos = [0.0, 0.0, 0.0]
        self._pos_lock = threading.Lock()
        self.odom_ok = False
        self.goal_cell = None
        self._recent_goals = []
        self._goals_reached_count = 0
        self.recovery_mode = False
        self._last_recovery_start_cell = None
        self.recovery_sub = self.create_subscription(
            Bool, "/route/recovery_mode", self._cb_recovery_mode, 10
        )

        self.plan_pub = self.create_publisher(
            Path, "/route/recovery_plan", 10
        )
        self.timer = self.create_timer(0.5, self._loop)
    
        self.get_logger().info("RouteGoalNavigator baslatildi (A* kapali, sadece goal publish)")

    def _cb_set_stage(self, msg: Int32):
        self.current_stage = int(msg.data)
        self._recent_goals.clear()
        self.goal_cell = None
        self._goals_reached_count = 0
        self.get_logger().info(f"Stage degisti → {self.current_stage}")

    def _cb_recovery_mode(self, msg: Bool):
        self.recovery_mode = bool(msg.data)
        self._last_recovery_start_cell = None

        if self.recovery_mode:
            self.get_logger().warn("Recovery mode aktif: 0,0 icin A* plan yayinlanacak.")
        else:
            self.get_logger().info("Recovery mode kapandi: normal goal publish devam.")

    def _find_path_cells(self, start_cell, goal_cell):
        from collections import deque

        queue = deque([start_cell])
        parent = {start_cell: None}

        while queue:
            r, c = queue.popleft()

            if (r, c) == goal_cell:
                break

            if self.walls is None:
                neighbors = [
                    (r - 1, c),
                    (r + 1, c),
                    (r, c + 1),
                    (r, c - 1),
                ]
            else:
                cell = self.walls[r][c]
                neighbors = []
                if not cell.get("N", False):
                    neighbors.append((r - 1, c))
                if not cell.get("S", False):
                    neighbors.append((r + 1, c))
                if not cell.get("E", False):
                    neighbors.append((r, c + 1))
                if not cell.get("W", False):
                    neighbors.append((r, c - 1))

            for nr, nc in neighbors:
                if not (0 <= nr < ROWS and 0 <= nc < COLS):
                    continue
                if (nr, nc) in parent:
                    continue
                if not self._is_cell_usable(nr, nc):
                    continue

                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))

        if goal_cell not in parent:
            return []

        path = []
        cur = goal_cell
        while cur is not None:
            path.append(cur)
            cur = parent[cur]

        path.reverse()
        return path

    def _publish_recovery_plan(self, start_cell):
        spawn_cell = DRONE_SPAWN_CELL

        path_cells = self._find_path_cells(start_cell, spawn_cell)

        if not path_cells:
            self.get_logger().error("Recovery icin 0,0 path bulunamadi.")
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for cell in path_cells:
            x, y = _cell_to_world(*cell)

            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = GOAL_Z
            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        self.plan_pub.publish(path_msg)

        self.get_logger().warn(
            f"Recovery /plan yayinlandi: {len(path_cells)} waypoint, hedef spawn/0,0"
        )


    def _cb_odom(self, msg: Odometry):
        with self._pos_lock:
            self.pos[0] = msg.pose.pose.position.x
            self.pos[1] = msg.pose.pose.position.y
            self.pos[2] = msg.pose.pose.position.z
        self.odom_ok = True

    def _is_cell_usable(self, r, c):
        if not (0 <= r < ROWS and 0 <= c < COLS):
            return False
        if self.walls is None:
            return True
        cell = self.walls[r][c]
        return not (cell.get("N", False) and cell.get("S", False) and cell.get("E", False) and cell.get("W", False))
    
    def _reachable_cells(self, start_cell):
        """BFS ile start_cell'den ulaşılabilen tüm hücreleri döndürür."""
        from collections import deque
        visited = set()
        queue = deque([start_cell])
        visited.add(start_cell)

        while queue:
            r, c = queue.popleft()
            if self.walls is None:
                # Duvar bilgisi yoksa her hücre erişilebilir say
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if (0 <= nr < ROWS and 0 <= nc < COLS 
                            and (nr, nc) not in visited):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
                continue

            cell = self.walls[r][c]
            # Her yönde: duvar yoksa komşuya geç
            neighbors = [
                ("N", r-1, c),
                ("S", r+1, c),
                ("E", r,   c+1),
                ("W", r,   c-1),
            ]
            for direction, nr, nc in neighbors:
                if (0 <= nr < ROWS and 0 <= nc < COLS
                        and not cell.get(direction, False)
                        and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return visited

    def _pick_goal_cell(self, curr_cell):
        r, c = curr_cell

        neighbors = [
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1),
        ]

        valid = []
        for nr, nc in neighbors:
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if not self._is_cell_usable(nr, nc):
                continue
            valid.append((nr, nc))

        if not valid:
            return curr_cell

        return random.choice(valid)

    def _publish_goal(self, goal_cell):
        gx, gy = _cell_to_world(*goal_cell)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = gx
        msg.pose.position.y = gy
        msg.pose.position.z = GOAL_Z
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def _loop(self):
        if not self.odom_ok:
            return
        with self._pos_lock:
            x, y = self.pos[0], self.pos[1]
        curr_cell = _world_to_cell(x, y)
        if self.recovery_mode:
            if curr_cell != self._last_recovery_start_cell:
                self._publish_recovery_plan(curr_cell)
                self._last_recovery_start_cell = curr_cell
            return

        need_new_goal = self.goal_cell is None
        if self.goal_cell is not None:
            gx, gy = _cell_to_world(*self.goal_cell)
            reached = math.hypot(x - gx, y - gy) < ARRIVAL_DIST
            if reached:
                self._goals_reached_count += 1
            need_new_goal = reached

        if need_new_goal:
            new_goal = self._pick_goal_cell(curr_cell)
            if new_goal is None:
                return
            self.goal_cell = new_goal
            self._recent_goals.append(new_goal)
            self._recent_goals = self._recent_goals[-RECENT_GOAL_MEMORY:]

        self._publish_goal(self.goal_cell)


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
