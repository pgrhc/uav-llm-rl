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

        self._review_in_progress = True
        threading.Thread(
            target=self.review_plan_interactively,
            args=(plan,),
            daemon=True
        ).start()

    def review_plan_interactively(self, plan: Dict[str, Any]):
        with self._prompt_lock:
            try:
                self.print_plan(plan)
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
        while rclpy.ok():
            user_input = input("Approve or Reject? [a/r]: ").strip().lower()
            if user_input in ["a", "approve"]:
                return "a"
            if user_input in ["r", "reject"]:
                return "r"
            print("Please enter 'a' to approve or 'r' to reject.")

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