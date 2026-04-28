#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LLMSceneSummaryNode(Node):
    def __init__(self):
        super().__init__("llm_scene_summary_node")

        self.latest_scene_graph: Optional[Dict[str, Any]] = None
        self.latest_threat_info: Optional[Dict[str, Any]] = None

        self.max_confirmed = 8
        self.max_candidate = 5
        self.max_lost = 4

        self.create_subscription(
            String,
            "/world/scene_graph",
            self.scene_graph_callback,
            10,
        )

        self.create_subscription(
            String,
            "/threat/target_info",
            self.threat_info_callback,
            10,
        )

        self.pub = self.create_publisher(
            String,
            "/llm/scene_summary",
            10,
        )

        self.timer = self.create_timer(0.5, self.publish_summary)

        self.get_logger().info("✅ llm_scene_summary_node started")

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def scene_graph_callback(self, msg: String):
        try:
            self.latest_scene_graph = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"scene_graph parse error: {e}")

    def threat_info_callback(self, msg: String):
        try:
            self.latest_threat_info = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"threat_info parse error: {e}")

    def format_object(self, obj: dict) -> dict:
        return {
            "object_id": obj.get("object_id"),
            "label": obj.get("label", "unknown"),
            "confidence": obj.get("confidence"),
            "position_2d": obj.get("position_2d"),
            "range_m": obj.get("range_m"),
            "bearing_rad": obj.get("bearing_rad"),
            "direction": obj.get("direction"),
            "area": obj.get("area", "unknown"),
            "state": obj.get("state", "unknown"),
        }

    def build_object_summary(self) -> dict:
        if not self.latest_scene_graph:
            return {
                "available": False,
                "confirmed_objects": [],
                "candidate_objects": [],
                "lost_objects": [],
                "ghost_count": 0,
                "nearest_confirmed_object": None,
            }

        sg = self.latest_scene_graph

        confirmed = sg.get("confirmed_objects", []) or []
        candidates = sg.get("candidate_objects", []) or []
        lost = sg.get("lost_objects", []) or []
        ghosts = sg.get("ghost_objects", []) or []

        confirmed_sorted = sorted(
            confirmed,
            key=lambda x: float(x.get("range_m") or 999.0)
        )

        candidate_sorted = sorted(
            candidates,
            key=lambda x: float(x.get("confidence") or 0.0),
            reverse=True,
        )

        lost_sorted = sorted(
            lost,
            key=lambda x: str(x.get("last_seen", "")),
            reverse=True,
        )

        nearest_confirmed = None
        if confirmed_sorted:
            nearest_confirmed = self.format_object(confirmed_sorted[0])

        return {
            "available": True,
            "object_count": sg.get("object_count", 0),
            "confirmed_count": sg.get("confirmed_count", len(confirmed)),
            "candidate_count": sg.get("candidate_count", len(candidates)),
            "lost_count": sg.get("lost_count", len(lost)),
            "ghost_count": sg.get("ghost_count", len(ghosts)),
            "confirmed_objects": [
                self.format_object(o) for o in confirmed_sorted[:self.max_confirmed]
            ],
            "candidate_objects": [
                self.format_object(o) for o in candidate_sorted[:self.max_candidate]
            ],
            "lost_objects": [
                self.format_object(o) for o in lost_sorted[:self.max_lost]
            ],
            "nearest_confirmed_object": nearest_confirmed,
        }

    def risk_level_from_score(self, score: float) -> str:
        if score >= 0.65:
            return "high"
        if score >= 0.35:
            return "medium"
        if score > 0.0:
            return "low"
        return "none"

    def build_threat_summary(self) -> dict:
        if not self.latest_threat_info:
            return {
                "available": False,
                "risk_level": "unknown",
                "max_risk": 0.0,
                "primary_threat": None,
                "target_scores": [],
            }

        info = self.latest_threat_info

        scores = info.get("target_scores", []) or []
        threats = info.get("top_threats", []) or []

        max_risk = 0.0
        if scores:
            max_risk = max(float(x) for x in scores)

        primary = None
        if threats:
            primary_raw = max(
                threats,
                key=lambda x: float(x.get("target_risk", 0.0))
            )

            primary = {
                "slot": primary_raw.get("slot"),
                "class_name": primary_raw.get("class_name", "Unknown"),
                "distance_m": primary_raw.get("dist"),
                "closing_speed": primary_raw.get("closing_speed"),
                "bearing_rad": primary_raw.get("bearing_rad"),
                "confidence": primary_raw.get("confidence"),
                "raw_risk": primary_raw.get("raw_risk"),
                "target_risk": primary_raw.get("target_risk"),
            }

        return {
            "available": True,
            "risk_level": self.risk_level_from_score(max_risk),
            "max_risk": round(max_risk, 4),
            "target_scores": scores,
            "primary_threat": primary,
        }

    def build_summary(self) -> dict:
        object_summary = self.build_object_summary()
        threat_summary = self.build_threat_summary()

        return {
            "timestamp": self.now_iso(),
            "scene_memory": object_summary,
            "dynamic_threats": threat_summary,
            "planner_hint": self.build_planner_hint(object_summary, threat_summary),
        }

    def build_planner_hint(self, object_summary: dict, threat_summary: dict) -> dict:
        hints: List[str] = []

        nearest = object_summary.get("nearest_confirmed_object")
        if nearest:
            hints.append(
                f"Nearest confirmed object is {nearest.get('label')} "
                f"at {nearest.get('range_m')} m, direction {nearest.get('direction')}."
            )

        risk_level = threat_summary.get("risk_level", "unknown")
        max_risk = threat_summary.get("max_risk", 0.0)

        if risk_level in ["medium", "high"]:
            hints.append(
                f"Dynamic threat risk is {risk_level} with max risk {max_risk}; "
                "planner should prefer cautious actions."
            )
        elif risk_level == "low":
            hints.append(
                f"Dynamic threat risk is low with max risk {max_risk}."
            )

        primary = threat_summary.get("primary_threat")
        if primary:
            hints.append(
                f"Primary dynamic threat: {primary.get('class_name')} "
                f"at {primary.get('distance_m')} m."
            )

        return {
            "safe_for_normal_navigation": risk_level in ["none", "low", "unknown"],
            "needs_caution": risk_level in ["medium", "high"],
            "text_hints": hints,
        }

    def publish_summary(self):
        payload = self.build_summary()

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LLMSceneSummaryNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()