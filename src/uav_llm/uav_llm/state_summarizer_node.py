import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import json
import math
import numpy as np

# --- STANDART MESAJLAR ---
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Float32MultiArray
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String, Float32MultiArray
from px4_msgs.msg import VehicleStatus

# --- VISION MESAJLARI (YOLO) ---
from vision_msgs.msg import Detection2DArray

# --- CUSTOM MESAJLAR ---
try:
    from fusion_msgs.msg import RadarPoints
except ImportError:
    pass



class StateSummarizerNode(Node):
    def __init__(self):
        super().__init__('state_summarizer_node')
        self.K = 5
        self.token_len = 7
        self.state_dim = 74
        self.lidar_sectors = 36
        self.max_speed = 5.0
        self.lidar_max_range = 30.0

        self.current_pos = None  
        self.current_yaw = 0.0
        self.odom_ok = False

        self.current_path = []
        self.initial_goal_distance = None

        self.state = {
            "speed": 0.0,
            "yaw": 0.0,
            "lidar": {
                "front_space": 99.0,
                "left_space": 99.0,
                "right_space": 99.0
            },
            "tracked_objects": [],
            "primary_threat": {
                "class": "None",
                "dist": 0.0,
                "risk_score": 0.0,
                "id": -1
            },
            "target_scores": [0.0] * self.K,
            "mission": {
                "path_available": False,
                "path_length_points": 0,
                "next_waypoint": None,
                "final_goal": None,
                "distance_to_next_waypoint": None,
                "distance_to_final_goal": None,
                "bearing_to_next_waypoint": None,
                "goal_direction": "unknown",
                "mission_progress": 0.0,
                "path_progress_ratio": 0.0,
                "route_deviation": None,
                "planner_status": "idle"
            },
            "vehicle": {
                "arming_state": 0,
                "nav_state": 0,
                "failsafe": False,
                "pre_flight_checks_pass": False,
                "gcs_connection_lost": False
            }
        }

        self.class_names = {
            0: "Unknown",
            1: "Drone",
            2: "Bird",
            3: "FixedWing",
            4: "Person"
        }

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(Float32MultiArray, '/threat/state_vec', self.state_vec_callback, 10)
        self.create_subscription(String, '/threat/target_info', self.target_info_callback, 10)
        self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            qos_be
        )

        self.create_subscription(
            Path,
            '/plan',
            self.plan_callback,
            qos_rel
        )

        self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            qos_be
        )

        self.summary_pub = self.create_publisher(String, '/llm/system_summary', 10)
        self.timer = self.create_timer(0.4, self.publish_summary)

        self.get_logger().info("✅ State Summarizer Başlatıldı")

    def vehicle_status_callback(self, msg: VehicleStatus):
        self.state["vehicle"]["arming_state"] = msg.arming_state
        self.state["vehicle"]["nav_state"] = msg.nav_state
        self.state["vehicle"]["failsafe"] = bool(msg.failsafe)
        self.state["vehicle"]["pre_flight_checks_pass"] = bool(msg.pre_flight_checks_pass)
        self.state["vehicle"]["gcs_connection_lost"] = bool(msg.gcs_connection_lost)

    def state_vec_callback(self, msg: Float32MultiArray):
        try:
            data = np.array(msg.data, dtype=np.float32)

            if data.shape[0] != self.state_dim:
                self.get_logger().warn(
                    f"Beklenen state dim={self.state_dim}, gelen={data.shape[0]}"
                )
                return
            speed_norm = float(data[0])
            yaw_sin = float(data[1])
            yaw_cos = float(data[2])

            self.state["speed"] = round(speed_norm * self.max_speed, 2)
            self.state["yaw"] = round(math.atan2(yaw_sin, yaw_cos), 3)
            lidar_norm = data[3:39]
            lidar_m = (lidar_norm * self.lidar_max_range).tolist()
            right_vals = lidar_m[0:12]
            front_vals = lidar_m[12:24]
            left_vals  = lidar_m[24:36]

            self.state["lidar"]["right_space"] = round(float(min(right_vals)) if right_vals else 99.0, 1)
            self.state["lidar"]["front_space"] = round(float(min(front_vals)) if front_vals else 99.0, 1)
            self.state["lidar"]["left_space"]  = round(float(min(left_vals)) if left_vals else 99.0, 1)
            objects_flat = data[39:]
            tracked_objects = []

            for i in range(self.K):
                start = i * self.token_len
                obj = objects_flat[start:start + self.token_len]

                class_id = int(obj[0])
                dist = float(obj[1])
                closing_speed = float(obj[2])
                bearing_sin = float(obj[3])
                bearing_cos = float(obj[4])
                confidence = float(obj[5])
                is_valid = float(obj[6])

                if is_valid <= 0.5:
                    continue

                bearing_rad = float(math.atan2(bearing_sin, bearing_cos))

                tracked_objects.append({
                    "slot": i,
                    "class_id": class_id,
                    "class": self.class_names.get(class_id, "Unknown"),
                    "dist": round(dist, 3),
                    "closing_speed": round(closing_speed, 3),
                    "bearing_rad": round(bearing_rad, 3),
                    "confidence": round(confidence, 3)
                })

            self.state["tracked_objects"] = tracked_objects

        except Exception as e:
            self.get_logger().warn(f"state_vec_callback hatası: {e}")


    def target_info_callback(self, msg: String):
        try:
            info = json.loads(msg.data)
            if "target_scores" in info and isinstance(info["target_scores"], list):
                scores = info["target_scores"][:self.K]
                while len(scores) < self.K:
                    scores.append(0.0)
                self.state["target_scores"] = [round(float(x), 4) for x in scores]
            top_threats = info.get("top_threats", [])
            if isinstance(top_threats, list) and len(top_threats) > 0:
                best = max(top_threats, key=lambda x: x.get("target_risk", 0.0))

                self.state["primary_threat"] = {
                    "class": best.get("class_name", "Unknown"),
                    "dist": round(float(best.get("dist", 0.0)), 1),
                    "risk_score": round(float(best.get("target_risk", 0.0)), 3),
                    "id": int(best.get("slot", -1))
                }
            else:
                self.reset_primary_threat()

        except json.JSONDecodeError:
            self.get_logger().warn("target_info JSON parse hatası!")
        except Exception as e:
            self.get_logger().warn(f"target_info_callback hatası: {e}")

    def odom_callback(self, msg: Odometry):
        try:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z
            self.current_pos = [x, y, z]
            self.odom_ok = True

            q = msg.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        except Exception as e:
            self.get_logger().warn(f"odom_callback hatası: {e}")

    def plan_callback(self, msg: Path):
        try:
            self.current_path = msg.poses

            mission = self.state["mission"]
            mission["path_available"] = len(self.current_path) > 0
            mission["path_length_points"] = len(self.current_path)

            if len(self.current_path) == 0:
                mission["next_waypoint"] = None
                mission["final_goal"] = None
                mission["distance_to_next_waypoint"] = None
                mission["distance_to_final_goal"] = None
                mission["bearing_to_next_waypoint"] = None
                mission["goal_direction"] = "unknown"
                mission["mission_progress"] = 0.0
                mission["path_progress_ratio"] = 0.0
                mission["route_deviation"] = None
                mission["planner_status"] = "idle"
                self.initial_goal_distance = None
                return

            final_pose = self.current_path[-1].pose.position
            mission["final_goal"] = {
                "x": round(final_pose.x, 3),
                "y": round(final_pose.y, 3),
                "z": round(final_pose.z, 3)
            }

            if self.odom_ok and self.current_pos is not None:
                final_dist = self.euclidean_distance_3d(
                    self.current_pos[0], self.current_pos[1], self.current_pos[2],
                    final_pose.x, final_pose.y, final_pose.z
                )

                if self.initial_goal_distance is None or final_dist > self.initial_goal_distance + 2.0:
                    self.initial_goal_distance = final_dist

            mission["planner_status"] = "navigating"

        except Exception as e:
            self.get_logger().warn(f"plan_callback hatası: {e}")


    def reset_primary_threat(self):
        self.state["primary_threat"] = {
            "class": "None",
            "dist": 0.0,
            "risk_score": 0.0,
            "id": -1
        }

    def euclidean_distance_2d(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def euclidean_distance_3d(self, x1, y1, z1, x2, y2, z2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def direction_from_relative_bearing(self, rel_bearing):
        deg = math.degrees(rel_bearing)

        if -20.0 <= deg <= 20.0:
            return "front"
        elif 20.0 < deg <= 70.0:
            return "front-left"
        elif -70.0 <= deg < -20.0:
            return "front-right"
        elif deg > 70.0:
            return "left"
        elif deg < -70.0:
            return "right"
        return "unknown"

    def find_next_waypoint_index(self):
        if not self.odom_ok or self.current_pos is None or len(self.current_path) == 0:
            return None

        if len(self.current_path) == 1:
            return 0

        curr_x, curr_y, _ = self.current_pos

        closest_idx = 0
        closest_dist = float('inf')

        for i, pose_stamped in enumerate(self.current_path):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            d = self.euclidean_distance_2d(curr_x, curr_y, px, py)

            if d < closest_dist:
                closest_dist = d
                closest_idx = i

        next_idx = min(closest_idx + 1, len(self.current_path) - 1)
        return next_idx

    def compute_route_deviation(self):
        if not self.odom_ok or self.current_pos is None or len(self.current_path) == 0:
            return None

        curr_x, curr_y, _ = self.current_pos
        min_dist = float('inf')

        for pose_stamped in self.current_path:
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            d = self.euclidean_distance_2d(curr_x, curr_y, px, py)
            if d < min_dist:
                min_dist = d

        return round(min_dist, 3)

    def update_mission_summary(self):
        mission = self.state["mission"]

        if len(self.current_path) == 0:
            mission["path_available"] = False
            mission["path_length_points"] = 0
            mission["next_waypoint"] = None
            mission["final_goal"] = None
            mission["distance_to_next_waypoint"] = None
            mission["distance_to_final_goal"] = None
            mission["bearing_to_next_waypoint"] = None
            mission["goal_direction"] = "unknown"
            mission["mission_progress"] = 0.0
            mission["path_progress_ratio"] = 0.0
            mission["route_deviation"] = None
            mission["planner_status"] = "idle"
            return

        mission["path_available"] = True
        mission["path_length_points"] = len(self.current_path)

        final_pose = self.current_path[-1].pose.position
        mission["final_goal"] = {
            "x": round(final_pose.x, 3),
            "y": round(final_pose.y, 3),
            "z": round(final_pose.z, 3)
        }

        if not self.odom_ok or self.current_pos is None:
            mission["planner_status"] = "waiting_for_odometry"
            return

        next_idx = self.find_next_waypoint_index()
        if next_idx is None:
            mission["planner_status"] = "waiting_for_path"
            return

        next_pose = self.current_path[next_idx].pose.position

        mission["next_waypoint"] = {
            "x": round(next_pose.x, 3),
            "y": round(next_pose.y, 3),
            "z": round(next_pose.z, 3)
        }

        curr_x, curr_y, curr_z = self.current_pos

        dist_next = self.euclidean_distance_3d(
            curr_x, curr_y, curr_z,
            next_pose.x, next_pose.y, next_pose.z
        )

        dist_final = self.euclidean_distance_3d(
            curr_x, curr_y, curr_z,
            final_pose.x, final_pose.y, final_pose.z
        )

        bearing_world = math.atan2(next_pose.y - curr_y, next_pose.x - curr_x)
        rel_bearing = self.normalize_angle(bearing_world - self.current_yaw)

        mission["distance_to_next_waypoint"] = round(dist_next, 3)
        mission["distance_to_final_goal"] = round(dist_final, 3)
        mission["bearing_to_next_waypoint"] = round(rel_bearing, 3)
        mission["goal_direction"] = self.direction_from_relative_bearing(rel_bearing)

        route_dev = self.compute_route_deviation()
        mission["route_deviation"] = route_dev

        if self.initial_goal_distance is not None and self.initial_goal_distance > 1e-6:
            progress = 1.0 - (dist_final / self.initial_goal_distance)
            progress = max(0.0, min(1.0, progress))
            mission["mission_progress"] = round(progress, 3)
        else:
            mission["mission_progress"] = 0.0
        if len(self.current_path) > 1:
            path_progress_ratio = next_idx / (len(self.current_path) - 1)
            mission["path_progress_ratio"] = round(path_progress_ratio, 3)
        else:
            mission["path_progress_ratio"] = 1.0

        if dist_final < 0.8:
            mission["planner_status"] = "goal_reached"
        elif route_dev is not None and route_dev > 2.0:
            mission["planner_status"] = "off_path"
        else:
            mission["planner_status"] = "navigating"


    def publish_summary(self, event=None):
        self.update_mission_summary()
        msg = String()
        msg.data = json.dumps(self.state)
        self.summary_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = StateSummarizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()