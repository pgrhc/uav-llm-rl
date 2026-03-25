#!/usr/bin/env python3

import json
import threading
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PlanReviewUINode(Node):
    def __init__(self):
        super().__init__("plan_review_ui")

        self.parsed_plan_sub = self.create_subscription(
            String,
            "/llm/parsed_plan",
            self.parsed_plan_callback,
            10
        )

        self.approved_plan_pub = self.create_publisher(
            String,
            "/llm/approved_plan",
            10
        )

        self.rejected_plan_pub = self.create_publisher(
            String,
            "/llm/rejected_plan",
            10
        )

        self.execution_status_pub = self.create_publisher(
            String,
            "/llm/execution_status",
            10
        )

        self._prompt_lock = threading.Lock()
        self._review_in_progress = False

        self.get_logger().info("plan_review_ui started. Listening on /llm/parsed_plan")

    def parsed_plan_callback(self, msg: String):
        if self._review_in_progress:
            self.get_logger().warning("A review is already in progress. New parsed plan ignored.")
            return

        try:
            plan = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse /llm/parsed_plan JSON: {e}")
            return

        is_safe, reason = self.deterministic_safety_check(plan)
        if not is_safe:
            self.get_logger().warn(f"Plan auto-rejected by safety gate: {reason}")
            if "safety_notes" not in plan:
                plan["safety_notes"] = []
            plan["safety_notes"].append(f"AUTO-REJECTED: {reason}")
            self.publish_rejected(plan)
            return

        self._review_in_progress = True
        threading.Thread(
            target=self.review_plan_interactively,
            args=(plan,),
            daemon=True
        ).start()

    def deterministic_safety_check(self, plan: Dict[str, Any]) -> tuple[bool, str]:
        steps = plan.get("steps", [])
        for step in steps:
            action = step.get("action", "")
            
            if action == "takeoff_to_altitude":
                alt = step.get("target_altitude", 0.0)
                if alt is not None and alt > 15.0:
                    return False, f"Target altitude {alt}m exceeds maximum allowed (15m)."
            
            if step.get("speed", 1.0) > 5.0:
                return False, f"Speed {step.get('speed')}m/s exceeds maximum allowed speed (5m/s)."
                
            if abs(step.get("delta_x", 0.0)) > 50.0 or abs(step.get("delta_y", 0.0)) > 50.0 or abs(step.get("delta_z", 0.0)) > 50.0:
                return False, "Move distance exceeds maximum allowed per step (50m)."
                
        return True, ""

    def review_plan_interactively(self, plan: Dict[str, Any]):
        with self._prompt_lock:
            try:
                self.print_plan(plan)
                
                if not plan.get("requires_approval", True):
                    print("\n--> Plan does not require approval. Auto-approving.")
                    self.publish_approved(plan)
                else:
                    decision = self.ask_decision()
    
                    if decision == "a":
                        self.publish_approved(plan)
                    else:
                        self.publish_rejected(plan)

            except Exception as e:
                self.get_logger().error(f"Review UI error: {e}")
            finally:
                self._review_in_progress = False

    def print_plan(self, plan: Dict[str, Any]):
        raw_command = plan.get("raw_command", "")
        reasoning = plan.get("reasoning", "")
        safety_notes = plan.get("safety_notes", [])
        requires_approval = plan.get("requires_approval", True)
        plan_id = plan.get("plan_id", "unknown_plan")
        steps = plan.get("steps", [])

        print("\n" + "=" * 72)
        print("PLAN REVIEW")
        print("=" * 72)
        print(f"Plan ID          : {plan_id}")
        print(f"Requires Approval: {requires_approval}")
        print(f"Raw Command      : {raw_command}")
        print("-" * 72)
        print("Extracted Plan:")
        if not steps:
            print("  (No steps found)")
        else:
            for idx, step in enumerate(steps, start=1):
                action = step.get("action", "unknown")
                step_id = step.get("step_id", idx)

                details = []
                for key in [
                    "target_altitude",
                    "delta_x",
                    "delta_y",
                    "delta_z",
                    "delta_yaw",
                    "speed"
                ]:
                    if key in step:
                        details.append(f"{key}={step[key]}")

                details_str = ", ".join(details) if details else "no extra parameters"
                print(f"  {step_id}. {action} ({details_str})")

        print("-" * 72)
        print(f"Reasoning: {reasoning if reasoning else '(none)'}")

        print("-" * 72)
        print("Safety Notes:")
        if isinstance(safety_notes, list) and safety_notes:
            for note in safety_notes:
                print(f"  - {note}")
        else:
            print("  (none)")
        print("=" * 72)

    def ask_decision(self) -> str:
        import sys
        while rclpy.ok():
            print("Approve or Reject? [a/r]: ", end="", flush=True)
            try:
                user_input = sys.stdin.readline()
                if not user_input:
                    # EOF hit, meaning stdin might not be forwarded by ros2 launch
                    self.get_logger().error("EOF on stdin. When running via launch file, stdin may not be attached nicely. Auto-rejecting.")
                    return "r"
                user_input = user_input.strip().lower()
            except Exception as e:
                self.get_logger().error(f"Input error: {e}")
                return "r"
                
            if user_input in ["a", "approve"]:
                return "a"
            if user_input in ["r", "reject"]:
                return "r"
            print("Please enter 'a' to approve or 'r' to reject.", flush=True)

        return "r"

    def publish_approved(self, plan: Dict[str, Any]):
        out = String()
        out.data = json.dumps(plan, ensure_ascii=False)
        self.approved_plan_pub.publish(out)

        status = {
            "plan_id": plan.get("plan_id", "unknown_plan"),
            "status": "approved",
            "message": "Plan approved by operator."
        }
        self.publish_execution_status(status)

        self.get_logger().info("Published approved plan to /llm/approved_plan")

    def publish_rejected(self, plan: Dict[str, Any]):
        out = String()
        out.data = json.dumps(plan, ensure_ascii=False)
        self.rejected_plan_pub.publish(out)

        status = {
            "plan_id": plan.get("plan_id", "unknown_plan"),
            "status": "rejected",
            "message": "Plan rejected by operator."
        }
        self.publish_execution_status(status)

        self.get_logger().info("Published rejected plan to /llm/rejected_plan")

    def publish_execution_status(self, status_dict: Dict[str, Any]):
        msg = String()
        msg.data = json.dumps(status_dict, ensure_ascii=False)
        self.execution_status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PlanReviewUINode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()