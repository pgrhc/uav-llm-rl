#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import String, Float32MultiArray
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray, Detection3DArray


class SemanticObjectBuilderNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_object_builder_node")

        self.state_dim = 74
        self.K = 5
        self.token_len = 7
        self.lidar_max_range = 30.0
        self.latest_lidar_36 = [self.lidar_max_range] * 36
        self.image_width = 1280.0
        self.image_height = 960.0
        self.camera_fov_deg = 90.0

        self.current_pos: Optional[List[float]] = None
        self.current_yaw: float = 0.0
        self.odom_ok = False

        self.latest_lidar_summary = {
            "front_space": None,
            "left_space": None,
            "right_space": None,
            "nearest_obstacle_m": None,
            "risk_level": "unknown",
        }
        self.class_names = {
        "0": "person",
        "1": "bicycle",
        "2": "car",
        "3": "motorcycle",
        "4": "airplane",
        "5": "bus",
        "6": "train",
        "7": "truck",
        "8": "boat",
        "9": "traffic_light",
        "10": "fire_hydrant",
        "11": "stop_sign",
        "12": "parking_meter",
        "13": "bench",
        "14": "bird",
        "15": "cat",
        "16": "dog",
        "17": "horse",
        "18": "sheep",
        "19": "cow",
        "20": "elephant",
        "21": "bear",
        "22": "zebra",
        "23": "giraffe",
        "24": "backpack",
        "25": "umbrella",
        "26": "handbag",
        "27": "tie",
        "28": "suitcase",
        "29": "frisbee",
        "30": "skis",
        "31": "snowboard",
        "32": "sports_ball",
        "33": "kite",
        "34": "baseball_bat",
        "35": "baseball_glove",
        "36": "skateboard",
        "37": "surfboard",
        "38": "tennis_racket",
        "39": "bottle",
        "40": "wine_glass",
        "41": "cup",
        "42": "fork",
        "43": "knife",
        "44": "spoon",
        "45": "bowl",
        "46": "banana",
        "47": "apple",
        "48": "sandwich",
        "49": "orange",
        "50": "broccoli",
        "51": "carrot",
        "52": "hot_dog",
        "53": "pizza",
        "54": "donut",
        "55": "cake",
        "56": "chair",
        "57": "couch",
        "58": "potted_plant",
        "59": "bed",
        "60": "dining_table",
        "61": "toilet",
        "62": "tv",
        "63": "laptop",
        "64": "mouse",
        "65": "remote",
        "66": "keyboard",
        "67": "cell_phone",
        "68": "microwave",
        "69": "oven",
        "70": "toaster",
        "71": "sink",
        "72": "refrigerator",
        "73": "book",
        "74": "clock",
        "75": "vase",
        "76": "scissors",
        "77": "teddy_bear",
        "78": "hair_drier",
        "79": "toothbrush",
    }

        self.latest_threat_objects = []
        self.latest_bev_detections = []

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            Detection2DArray,
            "/yolo/detections",
            self.yolo_callback,
            10,
        )

        self.create_subscription(
            Float32MultiArray,
            "/threat/state_vec",
            self.state_vec_callback,
            10,
        )

        self.create_subscription(
            Detection3DArray,
            "/yolo/projected_detections",
            self.bev_callback,
            10,
        )

        self.create_subscription(
            Odometry,
            "/odometry/filtered",
            self.odom_callback,
            qos_be,
        )

        self.semantic_pub = self.create_publisher(
            String,
            "/perception/semantic_objects",
            10,
        )

        self.get_logger().info("✅ semantic_object_builder_node started")

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
    
    def bev_callback(self, msg: Detection3DArray) -> None:
        try:
            bev_objects = []

            for det in msg.detections:
                if len(det.results) == 0:
                    continue

                result = det.results[0]
                class_id = str(result.hypothesis.class_id)
                confidence = float(result.hypothesis.score)

                px = float(result.pose.pose.position.x)
                py = float(result.pose.pose.position.y)
                pz = float(result.pose.pose.position.z)

                label = self.class_names.get(class_id, f"class_{class_id}")
                range_bev = math.sqrt(px * px + py * py)

                bev_objects.append({
                    "class_id": class_id,
                    "label": label,
                    "confidence": confidence,
                    "position_base": [px, py, pz],
                    "range_m": range_bev,
                })

            self.latest_bev_detections = bev_objects

        except Exception as e:
            self.get_logger().warn(f"bev_callback error: {e}")


    def find_bev_match(self, label: str, confidence: float):
        if not self.latest_bev_detections:
            return None

        candidates = [
            obj for obj in self.latest_bev_detections
            if obj.get("label") == label
        ]

        if not candidates:
            return None

        best = min(
            candidates,
            key=lambda x: abs(float(x.get("confidence", 0.0)) - confidence)
        )

        return best


    def estimate_map_position_from_bev(self, bev_obj):
        if not self.odom_ok or self.current_pos is None or bev_obj is None:
            return None

        try:
            bx, by, _ = bev_obj["position_base"]

            x, y, _ = self.current_pos
            yaw = self.current_yaw

            map_x = x + bx * math.cos(yaw) - by * math.sin(yaw)
            map_y = y + bx * math.sin(yaw) + by * math.cos(yaw)

            return [round(map_x, 3), round(map_y, 3)]

        except Exception:
            return None


    def fuse_range_sources(self, lidar_range, bev_range):
        if lidar_range is None and bev_range is None:
            return None, "no_range"

        if lidar_range is None:
            return round(float(bev_range), 2), "bev_only"

        if bev_range is None:
            return round(float(lidar_range), 2), "lidar_only"

        diff = abs(lidar_range - bev_range)

        if diff < 1.0:
            fused = 0.6 * bev_range + 0.4 * lidar_range
            return round(float(fused), 2), "bev_lidar_fused"

        if lidar_range < bev_range:
            return round(float(lidar_range), 2), "lidar_corrected_bev"

        return round(float(bev_range), 2), "bev_projection_unverified"


    def odom_callback(self, msg: Odometry) -> None:
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
            self.get_logger().warn(f"odom_callback error: {e}")

    def state_vec_callback(self, msg: Float32MultiArray) -> None:
        try:
            data = np.array(msg.data, dtype=np.float32)

            if data.shape[0] != self.state_dim:
                self.get_logger().warn(
                    f"Expected state dim={self.state_dim}, got={data.shape[0]}"
                )
                return

            lidar_norm = data[3:39]
            lidar_m = lidar_norm * self.lidar_max_range
            self.latest_lidar_36 = lidar_m.tolist()

            right_vals = lidar_m[0:12]
            front_vals = lidar_m[12:24]
            left_vals = lidar_m[24:36]

            right_space = float(np.min(right_vals))
            front_space = float(np.min(front_vals))
            left_space = float(np.min(left_vals))
            nearest = float(np.min(lidar_m))

            if nearest < 0.7:
                risk = "high"
            elif nearest < 1.5:
                risk = "medium"
            else:
                risk = "low"

            self.latest_lidar_summary = {
                "front_space": round(front_space, 2),
                "left_space": round(left_space, 2),
                "right_space": round(right_space, 2),
                "nearest_obstacle_m": round(nearest, 2),
                "risk_level": risk,
            }

            objects_flat = data[39:]
            threat_objects = []

            for i in range(self.K):
                start = i * self.token_len
                obj = objects_flat[start:start + self.token_len]

                class_id = int(obj[0])
                dist = float(obj[1])
                closing_speed = float(obj[2])
                bearing_sin = float(obj[3])
                bearing_cos = float(obj[4])
                confidence = float(obj[5])
                valid = float(obj[6])

                if valid <= 0.5:
                    continue

                bearing_rad = math.atan2(bearing_sin, bearing_cos)

                threat_objects.append({
                    "slot": i,
                    "class_id": class_id,
                    "distance_m": round(dist, 3),
                    "closing_speed": round(closing_speed, 3),
                    "bearing_rad": round(bearing_rad, 3),
                    "confidence": round(confidence, 3),
                })

            self.latest_threat_objects = threat_objects

        except Exception as e:
            self.get_logger().warn(f"state_vec_callback error: {e}")

    def bbox_to_bearing(self, center_x: float) -> float:
        """
        Image x center -> approximate camera bearing.
        center image = 0 deg
        left = negative
        right = positive
        """
        normalized_x = (center_x - self.image_width / 2.0) / (self.image_width / 2.0)
        bearing_deg = normalized_x * (self.camera_fov_deg / 2.0)
        return math.radians(bearing_deg)

    def direction_from_bearing(self, bearing_rad: float) -> str:
        deg = math.degrees(bearing_rad)

        if -15 <= deg <= 15:
            return "front"
        elif 15 < deg <= 45:
            return "front-right"
        elif -45 <= deg < -15:
            return "front-left"
        elif deg > 45:
            return "right"
        else:
            return "left"

    
    def estimate_range_from_bbox_lidar(self, bbox_center_x, bbox_size_x):
        if not self.latest_lidar_36 or len(self.latest_lidar_36) != 36:
            return None

        x1 = bbox_center_x - bbox_size_x / 2.0
        x2 = bbox_center_x + bbox_size_x / 2.0

        def x_to_bearing_deg(x):
            rel = (x - self.image_width / 2.0) / (self.image_width / 2.0)
            return rel * (self.camera_fov_deg / 2.0)

        deg1 = x_to_bearing_deg(x1)
        deg2 = x_to_bearing_deg(x2)

        min_deg = min(deg1, deg2)
        max_deg = max(deg1, deg2)

        # Biraz tolerans ekle
        min_deg -= 5.0
        max_deg += 5.0

        sector_values = []

        for i in range(36):
            # front merkez index 18 kabulü
            sector_deg = (i - 18) * 10.0

            if min_deg <= sector_deg <= max_deg:
                d = self.latest_lidar_36[i]
                if 0.05 < d < self.lidar_max_range:
                    sector_values.append(d)

        if not sector_values:
            return None

        # Objeye en yakın yüzeyi almak için median değil min daha mantıklı
        return round(float(min(sector_values)), 2)


    def estimate_map_position(self, range_m: Optional[float], bearing_rad: float):
        if not self.odom_ok or self.current_pos is None or range_m is None:
            return None

        x, y, _ = self.current_pos
        global_angle = self.current_yaw + bearing_rad

        obj_x = x + range_m * math.cos(global_angle)
        obj_y = y + range_m * math.sin(global_angle)

        return [round(obj_x, 3), round(obj_y, 3)]

    def get_detection_label_and_score(self, detection):
        label = "unknown"
        confidence = 0.0

        try:
            if len(detection.results) > 0:
                result = detection.results[0]
                label = str(result.hypothesis.class_id)
                confidence = float(result.hypothesis.score)

        except Exception:
            pass

        label = self.class_names.get(label, f"class_{label}")
        return label, confidence

    def yolo_callback(self, msg: Detection2DArray) -> None:
        objects = []

        try:
            for det in msg.detections:
                label, confidence = self.get_detection_label_and_score(det)

                bbox = det.bbox
                center_x = float(bbox.center.position.x)
                center_y = float(bbox.center.position.y)
                size_x = float(bbox.size_x)
                size_y = float(bbox.size_y)

                bearing_rad = self.bbox_to_bearing(center_x)
                direction = self.direction_from_bearing(bearing_rad)
                lidar_range_m = self.estimate_range_from_bbox_lidar(
                    det.bbox.center.position.x,
                    det.bbox.size_x
                )

                bev_match = self.find_bev_match(label, confidence)
                bev_range_m = None
                bev_map_pos = None

                if bev_match is not None:
                    bev_range_m = round(float(bev_match["range_m"]), 2)
                    bev_map_pos = self.estimate_map_position_from_bev(bev_match)

                range_m, fusion_mode = self.fuse_range_sources(lidar_range_m, bev_range_m)

                if bev_map_pos is not None and fusion_mode in {
                    "bev_only",
                    "bev_lidar_fused",
                    "bev_projection_unverified",
                }:
                    map_pos = bev_map_pos
                else:
                    map_pos = self.estimate_map_position(range_m, bearing_rad)

                obj = {
                    "object_id": f"obj_{uuid.uuid4().hex[:8]}",
                    "label": label,
                    "confidence": round(confidence, 3),
                    "source": "yolo_lidar_bev_fused",
                    "bbox": {
                        "center_x": round(center_x, 2),
                        "center_y": round(center_y, 2),
                        "size_x": round(size_x, 2),
                        "size_y": round(size_y, 2),
                    },
                    "bearing_rad": round(bearing_rad, 3),
                    "bearing_deg": round(math.degrees(bearing_rad), 2),
                    "direction": direction,
                    "range_m": range_m,
                    "range_sources": {
                        "lidar_range_m": lidar_range_m,
                        "bev_range_m": bev_range_m,
                        "selected_range_m": range_m,
                        "fusion_mode": fusion_mode,
                    },
                    "map_position_2d": map_pos,
                    "area": "unknown",
                    "state": "candidate",
                    "last_seen": self.now_iso(),
                }

                objects.append(obj)

            payload = {
                "timestamp": self.now_iso(),
                "frame_id": msg.header.frame_id,
                "object_count": len(objects),
                "objects": objects,
                "local_navigation_context": self.latest_lidar_summary,
                "threat_objects": self.latest_threat_objects,
            }

            out = String()
            out.data = json.dumps(payload, ensure_ascii=False)
            self.semantic_pub.publish(out)

        except Exception as e:
            self.get_logger().warn(f"yolo_callback error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SemanticObjectBuilderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()