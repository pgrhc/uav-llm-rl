#!/usr/bin/env python3

import json
import re
import uuid
from dataclasses import dataclass, asdict, field
from typing import Optional, List

import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


@dataclass
class PlanStep:
    step_id: int
    action: str
    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_z: float = 0.0
    delta_yaw: float = 0.0
    target_altitude: Optional[float] = None
    speed: float = 1.0


@dataclass
class MissionPlan:
    plan_id: str
    raw_command: str
    steps: List[PlanStep] = field(default_factory=list)
    reasoning: str = ""
    safety_notes: List[str] = field(default_factory=list)
    requires_approval: bool = True


SYSTEM_PROMPT = """
You are a UAV mission planner.

Convert the user's natural language command into a structured JSON mission plan.

Allowed actions:
- arm
- disarm
- takeoff_to_altitude
- land
- hover
- move_forward
- move_backward
- move_left
- move_right
- ascend
- descend
- rotate_cw
- rotate_ccw
- return_home

Rules:
- Output JSON only.
- If the user asks for multiple actions, split them into ordered steps.
- If altitude is explicitly requested for takeoff, use target_altitude.
- If a movement amount is not given, use defaults:
  - horizontal move: 5 meters
  - vertical move: 5 meters
  - rotation: 45 degrees
  - speed: 1.0 m/s
- Do not average numeric ranges like 2-3. Use the first number only.
- If the command is safety-critical (arm, disarm, takeoff, land, return_home), set requires_approval to true.
- Include short reasoning in English.
- Include safety_notes as a JSON array.

Output format:
{
  "plan_id": "auto",
  "raw_command": "<original command>",
  "steps": [
    {
      "step_id": 1,
      "action": "<allowed_action>",
      "delta_x": 0.0,
      "delta_y": 0.0,
      "delta_z": 0.0,
      "delta_yaw": 0.0,
      "target_altitude": null,
      "speed": 1.0
    }
  ],
  "reasoning": "<short explanation>",
  "safety_notes": ["<note1>", "<note2>"],
  "requires_approval": true
}
"""


class LLMMissionPlanner:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        endpoint: str = "http://localhost:11434/api/generate",
    ):
        self.model = model
        self.endpoint = endpoint

        self.defaults = {
            "horizontal": 5.0,
            "vertical": 5.0,
            "rotation": 45.0,
            "speed": 1.0,
        }

    def _call_llm(self, user_command: str) -> str:
        payload = {
            "model": self.model,
            "prompt": f"{SYSTEM_PROMPT}\n\nCommand: \"{user_command}\"\nOutput:",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
            },
        }
        response = requests.post(self.endpoint, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["response"].strip()

    def _extract_json(self, raw: str) -> dict:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"JSON not found. Raw output:\n{raw}")
        return json.loads(match.group())


    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(
            r"\b(ascend|descend|forward|backward|back|right|left|up|down)(-?\d+(\.\d+)?)\b",
            r"\1 \2",
            text,
        )
        text = re.sub(
            r"\b(clockwise|counterclockwise|anticlockwise)(-?\d+(\.\d+)?)\b",
            r"\1 \2",
            text,
        )

        text = text.replace(",", " and ")
        text = re.sub(r"\s+", " ", text)
        return text

    def _extract_first_number_after(self, text: str, patterns: list[str], default_value: float) -> Optional[float]:
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                tail = text[m.end():].strip()
                num_match = re.match(r"(-?\d+(\.\d+)?)", tail)
                if num_match:
                    return float(num_match.group(1))
                return default_value
        return None

    def _extract_takeoff_altitude(self, text: str) -> Optional[float]:
        patterns = [
            r"\btake ?off to (\d+(\.\d+)?)\s*(meters|meter|m)?\b",
            r"\btake ?off (\d+(\.\d+)?)\s*(meters|meter|m)?\b",
            r"\bto (\d+(\.\d+)?)\s*(meters|meter|m)\b",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return float(m.group(1))
        return None

    def _make_plan_id(self) -> str:
        return f"plan_{uuid.uuid4().hex[:8]}"

    def _build_mission_plan(self, raw_command: str, steps: List[PlanStep], reasoning: str) -> MissionPlan:
        approval_actions = {"arm", "disarm", "takeoff_to_altitude", "land", "return_home"}

        requires_approval = any(step.action in approval_actions for step in steps)

        safety_notes = []
        if any(step.action == "arm" for step in steps):
            safety_notes.append("Arming requires operator approval.")
        if any(step.action == "disarm" for step in steps):
            safety_notes.append("Disarming requires operator approval.")
        if any(step.action == "takeoff_to_altitude" for step in steps):
            safety_notes.append("Takeoff requires operator approval.")
        if any(step.action == "land" for step in steps):
            safety_notes.append("Landing requires operator approval.")
        if any(step.action == "return_home" for step in steps):
            safety_notes.append("Return-to-home requires operator approval.")

        return MissionPlan(
            plan_id=self._make_plan_id(),
            raw_command=raw_command,
            steps=steps,
            reasoning=reasoning,
            safety_notes=safety_notes,
            requires_approval=requires_approval,
        )

    def _rule_based_plan(self, user_command: str) -> Optional[MissionPlan]:
        text = self._normalize_text(user_command)
        steps: List[PlanStep] = []
        step_id = 1

        if re.search(r"\barm\b", text) and not re.search(r"\bdisarm\b", text):
            steps.append(PlanStep(step_id=step_id, action="arm"))
            step_id += 1

        if re.search(r"\bdisarm\b", text):
            steps.append(PlanStep(step_id=step_id, action="disarm"))
            step_id += 1

        if re.search(r"\btake ?off\b", text):
            alt = self._extract_takeoff_altitude(text)
            if alt is None:
                alt = self.defaults["vertical"]
            steps.append(
                PlanStep(
                    step_id=step_id,
                    action="takeoff_to_altitude",
                    target_altitude=float(alt),
                    speed=self.defaults["speed"],
                )
            )
            step_id += 1

        if re.search(r"\bland\b", text):
            steps.append(PlanStep(step_id=step_id, action="land"))
            step_id += 1

        if re.search(r"\breturn home\b|\brth\b|\brtl\b|\bgo home\b", text):
            steps.append(PlanStep(step_id=step_id, action="return_home"))
            step_id += 1

        if re.search(r"\bhover\b|\bstop\b|\bwait\b", text):
            steps.append(PlanStep(step_id=step_id, action="hover"))
            step_id += 1

        val = self._extract_first_number_after(
            text,
            [r"\bascend\b", r"\bup\b", r"\bgo up\b", r"\brise\b"],
            self.defaults["vertical"],
        )
        if val is not None:
            steps.append(
                PlanStep(
                    step_id=step_id,
                    action="ascend",
                    delta_z=abs(val),
                    speed=self.defaults["speed"],
                )
            )
            step_id += 1

        val = self._extract_first_number_after(
            text,
            [r"\bdescend\b", r"\bdown\b", r"\bgo down\b"],
            self.defaults["vertical"],
        )
        if val is not None:
            steps.append(
                PlanStep(
                    step_id=step_id,
                    action="descend",
                    delta_z=-abs(val),
                    speed=self.defaults["speed"],
                )
            )
            step_id += 1

        val = self._extract_first_number_after(
            text,
            [r"\bmove forward\b", r"\bforward\b", r"\bgo forward\b"],
            self.defaults["horizontal"],
        )
        if val is not None:
            steps.append(
                PlanStep(
                    step_id=step_id,
                    action="move_forward",
                    delta_x=abs(val),
                    speed=self.defaults["speed"],
                )
            )
            step_id += 1

        val = self._extract_first_number_after(
            text,
            [r"\bmove backward\b", r"\bbackward\b", r"\bgo back\b", r"\bback\b"],
            self.defaults["horizontal"],
        )
        if val is not None:
            steps.append(
                PlanStep(
                    step_id=step_id,
                    action="move_backward",
                    delta_x=-abs(val),
                    speed=self.defaults["speed"],
                )
            )
            step_id += 1

        val = self._extract_first_number_after(
            text,
            [r"\bmove right\b", r"\bright\b", r"\bgo right\b"],
            self.defaults["horizontal"],
        )
        if val is not None:
            steps.append(
                PlanStep(
                    step_id=step_id,
                    action="move_right",
                    delta_y=abs(val),
                    speed=self.defaults["speed"],
                )
            )
            step_id += 1

        val = self._extract_first_number_after(
            text,
            [r"\bmove left\b", r"\bleft\b", r"\bgo left\b"],
            self.defaults["horizontal"],
        )
        if val is not None:
            steps.append(
                PlanStep(
                    step_id=step_id,
                    action="move_left",
                    delta_y=-abs(val),
                    speed=self.defaults["speed"],
                )
            )
            step_id += 1


        val = self._extract_first_number_after(
            text,
            [r"\brotate clockwise\b", r"\bclockwise\b", r"\bturn clockwise\b"],
            self.defaults["rotation"],
        )
        if val is not None:
            steps.append(
                PlanStep(
                    step_id=step_id,
                    action="rotate_cw",
                    delta_yaw=abs(val),
                    speed=self.defaults["speed"],
                )
            )
            step_id += 1

        val = self._extract_first_number_after(
            text,
            [r"\brotate counterclockwise\b", r"\bcounterclockwise\b", r"\banticlockwise\b", r"\bturn counterclockwise\b"],
            self.defaults["rotation"],
        )
        if val is not None:
            steps.append(
                PlanStep(
                    step_id=step_id,
                    action="rotate_ccw",
                    delta_yaw=-abs(val),
                    speed=self.defaults["speed"],
                )
            )
            step_id += 1

        if not steps:
            return None

        reasoning = "Mission plan parsed deterministically from the user's natural language command."
        return self._build_mission_plan(user_command, steps, reasoning)

    def _plan_from_llm(self, user_command: str) -> MissionPlan:
        raw = self._call_llm(user_command)
        cmd = self._extract_json(raw)

        steps = []
        for idx, step in enumerate(cmd.get("steps", []), start=1):
            steps.append(
                PlanStep(
                    step_id=int(step.get("step_id", idx)),
                    action=step.get("action", "hover"),
                    delta_x=float(step.get("delta_x", 0.0)),
                    delta_y=float(step.get("delta_y", 0.0)),
                    delta_z=float(step.get("delta_z", 0.0)),
                    delta_yaw=float(step.get("delta_yaw", 0.0)),
                    target_altitude=(
                        None if step.get("target_altitude", None) is None
                        else float(step.get("target_altitude"))
                    ),
                    speed=float(step.get("speed", self.defaults["speed"])),
                )
            )

        if not steps:
            raise ValueError("LLM produced no plan steps.")

        return MissionPlan(
            plan_id=cmd.get("plan_id", self._make_plan_id()),
            raw_command=cmd.get("raw_command", user_command),
            steps=steps,
            reasoning=cmd.get("reasoning", ""),
            safety_notes=cmd.get("safety_notes", []),
            requires_approval=bool(cmd.get("requires_approval", True)),
        )

    def plan(self, user_command: str) -> MissionPlan:
        mission_plan = self._rule_based_plan(user_command)
        if mission_plan is not None:
            return mission_plan
        return self._plan_from_llm(user_command)


class LLMTaskNode(Node):
    def __init__(self):
        super().__init__("llm_task_node")

        self.planner = LLMMissionPlanner()

        self.command_sub = self.create_subscription(
            String,
            "/llm/user_command_raw",
            self.command_callback,
            10,
        )

        self.parsed_plan_pub = self.create_publisher(
            String,
            "/llm/parsed_plan",
            10,
        )

        self.get_logger().info(
            "LLM task node started. Listening on /llm/user_command_raw and publishing /llm/parsed_plan"
        )

    def command_callback(self, msg: String):
        user_text = msg.data.strip()
        self.get_logger().info(f"Received raw command: {user_text}")

        try:
            plan = self.planner.plan(user_text)

            out = String()
            out.data = json.dumps(asdict(plan), ensure_ascii=False)
            self.parsed_plan_pub.publish(out)

            self.get_logger().info(f"Published parsed plan: {out.data}")

        except Exception as e:
            self.get_logger().error(f"Planning failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LLMTaskNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()