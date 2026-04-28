#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SemanticMemoryNode(Node):
    def __init__(self):
        super().__init__("semantic_memory_node")

        self.match_distance_m = 1.0
        self.confirm_seen_count = 15
        self.lost_after_sec = 2.0
        self.delete_after_sec = 10.0

        self.objects: Dict[str, dict] = {}

        self.sub = self.create_subscription(
            String,
            "/perception/semantic_objects",
            self.semantic_objects_callback,
            10,
        )

        self.pub = self.create_publisher(
            String,
            "/world/scene_graph",
            10,
        )

        self.timer = self.create_timer(0.5, self.publish_scene_graph)
        self.cross_label_duplicate_dist = 0.35
        self.get_logger().info("✅ semantic_memory_node started")

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def distance_2d(self, p1, p2) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def find_matching_object(self, new_obj: dict) -> Optional[str]:
        new_label = new_obj.get("label")
        new_pos = new_obj.get("map_position_2d")

        if new_label is None or new_pos is None:
            return None

        best_id = None
        best_dist = float("inf")

        for obj_id, old_obj in self.objects.items():
            if old_obj.get("label") != new_label:
                continue

            old_pos = old_obj.get("position_2d")
            if old_pos is None:
                continue

            d = self.distance_2d(new_pos, old_pos)

            if d < self.match_distance_m and d < best_dist:
                best_dist = d
                best_id = obj_id

        return best_id

    def semantic_objects_callback(self, msg: String):
        try:
            payload = json.loads(msg.data)
            detections = payload.get("objects", [])
            current_time = self.now_sec()

            for det in detections:
                label = det.get("label", "unknown")
                pos = det.get("map_position_2d")
                confidence = float(det.get("confidence", 0.0))
                range_m = det.get("range_m")
                bearing_rad = det.get("bearing_rad")
                direction = det.get("direction")
                source = det.get("source", "unknown")
                bbox = det.get("bbox", {})
                area = det.get("area", "unknown")
                range_sources = det.get("range_sources", {})

                if pos is None:
                    continue

                matched_id = self.find_matching_object(det)

                if matched_id is None:
                    object_id = f"mem_{uuid.uuid4().hex[:8]}"

                    self.objects[object_id] = {
                        "object_id": object_id,
                        "label": label,
                        "confidence": round(confidence, 3),
                        "position_2d": pos,
                        "range_m": range_m,
                        "bearing_rad": bearing_rad,
                        "direction": direction,
                        "bbox": bbox,
                        "area": area,
                        "source": source,
                        "range_sources": range_sources,
                        "state": "candidate",
                        "seen_count": 1,
                        "first_seen": self.now_iso(),
                        "last_seen": self.now_iso(),
                        "_last_seen_sec": current_time,
                    }

                else:
                    obj = self.objects[matched_id]

                    old_seen = int(obj.get("seen_count", 0))
                    new_seen = old_seen + 1

                    old_conf = float(obj.get("confidence", 0.0))
                    alpha = 0.6

                    old_pos = obj.get("position_2d", pos)
                    fused_pos = [
                        round(alpha * pos[0] + (1.0 - alpha) * old_pos[0], 3),
                        round(alpha * pos[1] + (1.0 - alpha) * old_pos[1], 3),
                    ]

                    obj["confidence"] = round(0.7 * confidence + 0.3 * old_conf, 3)
                    obj["position_2d"] = fused_pos
                    obj["range_m"] = range_m
                    obj["bearing_rad"] = bearing_rad
                    obj["direction"] = direction
                    obj["bbox"] = bbox
                    obj["area"] = area
                    obj["source"] = source
                    obj["range_sources"] = range_sources
                    obj["seen_count"] = new_seen
                    obj["last_seen"] = self.now_iso()
                    obj["_last_seen_sec"] = current_time

                    if new_seen >= self.confirm_seen_count:
                        obj["state"] = "confirmed"
                    else:
                        obj["state"] = "candidate"

        except Exception as e:
            self.get_logger().warn(f"semantic_objects_callback error: {e}")

    def update_lost_and_delete(self):
        now = self.now_sec()
        to_delete: List[str] = []

        for obj_id, obj in self.objects.items():
            last_seen = float(obj.get("_last_seen_sec", now))
            age = now - last_seen

            if age > self.delete_after_sec:
                to_delete.append(obj_id)
            elif age > self.lost_after_sec:
                if obj.get("state") == "confirmed":
                    obj["state"] = "lost"
                elif obj.get("state") == "candidate":
                    to_delete.append(obj_id)

        for obj_id in to_delete:
            del self.objects[obj_id]

    def publish_scene_graph(self):
        self.update_lost_and_delete()
        self.suppress_cross_label_duplicates()

        visible_objects = []
        confirmed_objects = []
        lost_objects = []
        candidate_objects = []
        ghost_objects = []

        for obj in self.objects.values():
            clean_obj = dict(obj)
            clean_obj.pop("_last_seen_sec", None)

            state = clean_obj.get("state")

            if state == "ghost":
                ghost_objects.append(clean_obj)
                continue

            visible_objects.append(clean_obj)

            if state == "confirmed":
                confirmed_objects.append(clean_obj)
            elif state == "lost":
                lost_objects.append(clean_obj)
            else:
                candidate_objects.append(clean_obj)

        payload = {
            "timestamp": self.now_iso(),
            "frame": "map",
            "object_count": len(visible_objects),
            "confirmed_count": len(confirmed_objects),
            "candidate_count": len(candidate_objects),
            "lost_count": len(lost_objects),
            "ghost_count": len(ghost_objects),
            "objects": visible_objects,
            "confirmed_objects": confirmed_objects,
            "candidate_objects": candidate_objects,
            "lost_objects": lost_objects,
            "ghost_objects": ghost_objects,
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub.publish(msg)


    def suppress_cross_label_duplicates(self):
        objs = list(self.objects.values())

        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                a = objs[i]
                b = objs[j]

                if a["state"] in ["lost", "ghost"] or b["state"] in ["lost", "ghost"]:
                    continue

                if a["label"] == b["label"]:
                    continue

                pa = a.get("position_2d")
                pb = b.get("position_2d")
                if pa is None or pb is None:
                    continue

                d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])

                if d < self.cross_label_duplicate_dist:
                    if a["confidence"] >= b["confidence"]:
                        b["state"] = "ghost"
                        b["ghost_reason"] = f"spatial_duplicate_of_{a['object_id']}"
                    else:
                        a["state"] = "ghost"
                        a["ghost_reason"] = f"spatial_duplicate_of_{b['object_id']}"


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMemoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()