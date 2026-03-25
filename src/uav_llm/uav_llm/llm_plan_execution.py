#!/usr/bin/env python3

import json
import math
import time
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ExecutorNode(Node):
    def __init__(self):
        super().__init__("executor_node")

        self.approved_plan_sub = self.create_subscription(
            String,
            "/llm/approved_plan",
            self.approved_plan_callback,
            10
        )
        self.feedback_sub = self.create_subscription(
            String,
            "/llm/execution_feedback",
            self.feedback_callback,
            10
        )
        self.command_pub = self.create_publisher(
            String,
            "/llm/parsed_command",
            10
        )

        self.execution_status_pub = self.create_publisher(
            String,
            "/llm/execution_status",
            10
        )

        self.active_plan: Optional[Dict[str, Any]] = None
        self.active_steps: List[Dict[str, Any]] = []
        self.current_step_index: int = -1
        self.step_deadline: Optional[float] = None
        self.plan_running: bool = False
        self.waiting_for_feedback: bool = False

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            "executor_node started. Listening on /llm/approved_plan"
        )

    def approved_plan_callback(self, msg: String):
        if self.plan_running:
            self.get_logger().warning(
                "A plan is already running. New approved plan ignored."
            )
            return

        try:
            plan = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse approved plan JSON: {e}")
            return

        steps = plan.get("steps", [])
        if not steps:
            self.get_logger().error("Approved plan contains no steps.")
            self.publish_status(
                plan_id=plan.get("plan_id", "unknown_plan"),
                status="failed",
                message="Approved plan contains no steps."
            )
            return

        self.active_plan = plan
        self.active_steps = steps
        self.current_step_index = -1
        self.step_deadline = None
        self.plan_running = True
        self.waiting_for_feedback = False

        self.get_logger().info(
            f"Approved plan received: {plan.get('plan_id', 'unknown_plan')} "
            f"with {len(steps)} step(s)."
        )

        self.publish_status(
            plan_id=plan.get("plan_id", "unknown_plan"),
            status="started",
            message="Execution started."
        )

        self.start_next_step()

    def timer_callback(self):
        if not self.plan_running or not self.waiting_for_feedback:
            return

        if self.step_deadline is None:
            return

        if time.time() >= self.step_deadline:
            plan_id = self.active_plan.get("plan_id", "unknown_plan") if self.active_plan else "unknown"
            self.get_logger().error(f"Step timeout reached.")
            self.publish_status(
                plan_id=plan_id,
                current_step=self.current_step_index + 1,
                status="failed",
                message="Step execution timed out."
            )
            self.reset_execution_state()

    def feedback_callback(self, msg: String):
        if not self.plan_running or not self.waiting_for_feedback:
            return

        try:
            feedback = json.loads(msg.data)
            status = feedback.get("status")

            if status == "success":
                self.waiting_for_feedback = False
                self.start_next_step()
            elif status == "failure":
                reason = feedback.get("reason", "unknown error")
                plan_id = self.active_plan.get("plan_id", "unknown_plan") if self.active_plan else "unknown"
                self.get_logger().error(f"Step failed from feedback: {reason}")
                self.publish_status(
                    plan_id=plan_id,
                    current_step=self.current_step_index + 1,
                    status="failed",
                    message=f"Step execution failed: {reason}"
                )
                self.reset_execution_state()
        except Exception as e:
            self.get_logger().error(f"Failed to parse feedback: {e}")

    def start_next_step(self):
        if self.active_plan is None:
            self.reset_execution_state()
            return

        self.current_step_index += 1

        if self.current_step_index >= len(self.active_steps):
            plan_id = self.active_plan.get("plan_id", "unknown_plan")
            self.publish_status(
                plan_id=plan_id,
                status="completed",
                message="Plan finished successfully."
            )
            self.get_logger().info(f"Plan completed: {plan_id}")
            self.reset_execution_state()
            return

        step = self.active_steps[self.current_step_index]
        plan_id = self.active_plan.get("plan_id", "unknown_plan")

        try:
            command_dict = self.step_to_command(step)
            estimated_duration = self.estimate_step_duration(step)

            out = String()
            out.data = json.dumps(command_dict, ensure_ascii=False)
            self.command_pub.publish(out)

            self.step_deadline = time.time() + max(30.0, estimated_duration * 2)
            self.waiting_for_feedback = True

            self.publish_status(
                plan_id=plan_id,
                current_step=self.current_step_index + 1,
                status="executing",
                message=self.make_step_status_message(step)
            )

            self.get_logger().info(
                f"Executing step {self.current_step_index + 1}/{len(self.active_steps)} | "
                f"action={step.get('action', 'unknown')} | "
                f"estimated_duration={estimated_duration:.2f}s"
            )

        except Exception as e:
            self.get_logger().error(f"Failed to execute step: {e}")
            self.publish_status(
                plan_id=plan_id,
                current_step=self.current_step_index + 1,
                status="failed",
                message=f"Step execution failed: {e}"
            )
            self.reset_execution_state()

    def step_to_command(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a mission-plan step into the low-level command format
        currently used by /llm/parsed_command.
        """

        action = step.get("action", "hover")
        speed = float(step.get("speed", 1.0))
        command = {
            "action": action,
            "delta_x": float(step.get("delta_x", 0.0)),
            "delta_y": float(step.get("delta_y", 0.0)),
            "delta_z": float(step.get("delta_z", 0.0)),
            "delta_yaw": float(step.get("delta_yaw", 0.0)),
            "speed": speed,
            "reasoning": f"Executing approved plan step: {action}"
        }
        if action == "takeoff_to_altitude":
            command["target_altitude"] = (
                None if step.get("target_altitude", None) is None
                else float(step.get("target_altitude"))
            )

        return command

    def estimate_step_duration(self, step: Dict[str, Any]) -> float:
        action = step.get("action", "hover")
        speed = max(0.1, float(step.get("speed", 1.0)))

        if action in ["arm", "disarm"]:
            return 2.0
            
        if action == "follow_path":
            return 120.0

        if action == "takeoff_to_altitude":
            target_altitude = step.get("target_altitude", 5.0)
            if target_altitude is None:
                target_altitude = 5.0
            return max(3.0, float(target_altitude) / speed)

        if action == "land":
            return 5.0

        if action == "return_home":
            return 8.0

        if action == "hover":
            return 2.0

        if action in ["ascend", "descend"]:
            dz = abs(float(step.get("delta_z", 0.0)))
            return max(1.0, dz / speed)

        if action in ["move_forward", "move_backward"]:
            dx = abs(float(step.get("delta_x", 0.0)))
            return max(1.0, dx / speed)

        if action in ["move_left", "move_right"]:
            dy = abs(float(step.get("delta_y", 0.0)))
            return max(1.0, dy / speed)

        if action in ["rotate_cw", "rotate_ccw"]:
            dyaw = abs(float(step.get("delta_yaw", 0.0)))
            yaw_rate_deg = 45.0  
            return max(1.0, dyaw / yaw_rate_deg)

        distance = max(
            abs(float(step.get("delta_x", 0.0))),
            abs(float(step.get("delta_y", 0.0))),
            abs(float(step.get("delta_z", 0.0))),
        )
        if distance > 0.0:
            return max(1.0, distance / speed)

        return 2.0

    def make_step_status_message(self, step: Dict[str, Any]) -> str:
        action = step.get("action", "unknown")

        if action == "arm":
            return "Arming vehicle"
            
        if action == "follow_path":
            return "Following waypoints from /plan"

        if action == "disarm":
            return "Disarming vehicle"

        if action == "takeoff_to_altitude":
            alt = step.get("target_altitude", "unknown")
            return f"Climbing to {alt} meters"

        if action == "land":
            return "Landing vehicle"

        if action == "return_home":
            return "Returning home"

        if action == "hover":
            return "Holding position"

        if action == "ascend":
            return f"Ascending by {abs(float(step.get('delta_z', 0.0))):.1f} meters"

        if action == "descend":
            return f"Descending by {abs(float(step.get('delta_z', 0.0))):.1f} meters"

        if action == "move_forward":
            return f"Moving forward by {abs(float(step.get('delta_x', 0.0))):.1f} meters"

        if action == "move_backward":
            return f"Moving backward by {abs(float(step.get('delta_x', 0.0))):.1f} meters"

        if action == "move_right":
            return f"Moving right by {abs(float(step.get('delta_y', 0.0))):.1f} meters"

        if action == "move_left":
            return f"Moving left by {abs(float(step.get('delta_y', 0.0))):.1f} meters"

        if action == "rotate_cw":
            return f"Rotating clockwise by {abs(float(step.get('delta_yaw', 0.0))):.1f} degrees"

        if action == "rotate_ccw":
            return f"Rotating counterclockwise by {abs(float(step.get('delta_yaw', 0.0))):.1f} degrees"

        return f"Executing {action}"

    def publish_status(
        self,
        plan_id: str,
        status: str,
        message: str,
        current_step: Optional[int] = None
    ):
        payload = {
            "plan_id": plan_id,
            "status": status,
            "message": message,
        }

        if current_step is not None:
            payload["current_step"] = current_step

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.execution_status_pub.publish(msg)

    def reset_execution_state(self):
        self.active_plan = None
        self.active_steps = []
        self.current_step_index = -1
        self.step_deadline = None
        self.plan_running = False
        self.waiting_for_feedback = False


def main(args=None):
    rclpy.init(args=args)
    node = ExecutorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()