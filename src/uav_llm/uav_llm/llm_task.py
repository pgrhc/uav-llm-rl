#!/usr/bin/env python3

import json
import re
from dataclasses import dataclass, asdict
from typing import Optional

import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


@dataclass
class DroneCommand:
    action: str
    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_z: float = 0.0
    delta_yaw: float = 0.0
    speed: float = 1.0
    reasoning: str = ""


SYSTEM_PROMPT = """
You are a UAV (drone) command interpreter.
Convert the user's natural language command into JSON.

DEFAULT VALUES (if user does not specify a value):
- Vertical movement (delta_z): 5 meters
- Horizontal movement (delta_x, delta_y): 5 meters
- Rotation (delta_yaw): 45 degrees
- Speed: 1.0 m/s

AXIS DEFINITIONS:
- delta_z > 0 : ascend
- delta_z < 0 : descend
- delta_x > 0 : move forward
- delta_x < 0 : move backward
- delta_y > 0 : move right
- delta_y < 0 : move left
- delta_yaw > 0 : rotate clockwise
- delta_yaw < 0 : rotate counter-clockwise

IMPORTANT RULES:
- Do NOT average ranges like "2-3".
- If a range appears, prefer the first number only.
- If the user writes a command with an attached number like "ascend2", interpret it as "ascend 2".
- If the user gives no number, use the default value for that action.
- If the command contains multiple actions, combine them into one JSON output.

OUTPUT FORMAT (JSON only, no extra text):
{
  "action": "<action_name>",
  "delta_x": <float>,
  "delta_y": <float>,
  "delta_z": <float>,
  "delta_yaw": <float>,
  "speed": <float>,
  "reasoning": "<short explanation in English>"
}
"""


class LLMCommandParser:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        endpoint: str = "http://localhost:11434/api/generate",
        defaults: Optional[dict] = None
    ):
        self.model = model
        self.endpoint = endpoint
        self.defaults = defaults or {
            "delta_z": 5.0,
            "delta_x": 5.0,
            "delta_y": 5.0,
            "delta_yaw": 45.0,
            "speed": 1.0
        }

        self.action_patterns = {
            "ascend": [
                r"\bascend\b", r"\bup\b", r"\bgo up\b", r"\brise\b"
            ],
            "descend": [
                r"\bdescend\b", r"\bdown\b", r"\bgo down\b"
            ],
            "move_forward": [
                r"\bmove forward\b", r"\bforward\b", r"\bgo forward\b"
            ],
            "move_backward": [
                r"\bmove backward\b", r"\bbackward\b", r"\bgo back\b", r"\bback\b"
            ],
            "move_right": [
                r"\bmove right\b", r"\bright\b", r"\bgo right\b"
            ],
            "move_left": [
                r"\bmove left\b", r"\bleft\b", r"\bgo left\b"
            ],
            "rotate_cw": [
                r"\brotate clockwise\b", r"\bclockwise\b", r"\bturn clockwise\b"
            ],
            "rotate_ccw": [
                r"\brotate counterclockwise\b", r"\bcounterclockwise\b",
                r"\banticlockwise\b", r"\bturn counterclockwise\b"
            ],
            "hover": [
                r"\bhover\b", r"\bstop\b", r"\bwait\b"
            ],
            "arm":         [r"\barm\b"],
            "disarm":      [r"\bdisarm\b"],
            "takeoff":     [r"\btakeoff\b", r"\btake off\b"],
            "land":        [r"\bland\b"],
            "return_home": [r"\breturn home\b", r"\brth\b", r"\brtl\b", r"\bgo home\b"],

        }

    def _call_llm(self, user_command: str) -> str:
        payload = {
            "model": self.model,
            "prompt": f"{SYSTEM_PROMPT}\n\nCommand: \"{user_command}\"\nOutput:",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
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
            text
        )
        text = re.sub(
            r"\b(clockwise|counterclockwise|anticlockwise)(-?\d+(\.\d+)?)\b",
            r"\1 \2",
            text
        )
        text = text.replace(",", " and ")
        return re.sub(r"\s+", " ", text)

    def _find_action_amount(self, text: str, patterns: list[str], default_value: float) -> Optional[float]:
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                tail = text[m.end():].strip()
                num_match = re.match(r"(-?\d+(\.\d+)?)", tail)
                if num_match:
                    return float(num_match.group(1))
                return default_value
        return None

    def _rule_based_parse(self, user_command: str) -> Optional[dict]:
        text = self._normalize_text(user_command)

        if re.fullmatch(r"\d+(\.\d+)?\s*-\s*\d+(\.\d+)?", text):
            raise ValueError("Ambiguous range command. Please provide a single value.")

        result = {
            "action": "hover",
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "delta_yaw": 0.0,
            "speed": self.defaults["speed"],
            "reasoning": ""
        }

        matched = []

        val = self._find_action_amount(text, self.action_patterns["ascend"], self.defaults["delta_z"])
        if val is not None:
            result["delta_z"] += abs(val)
            matched.append(f"ascend {abs(val)} m")

        val = self._find_action_amount(text, self.action_patterns["descend"], self.defaults["delta_z"])
        if val is not None:
            result["delta_z"] -= abs(val)
            matched.append(f"descend {abs(val)} m")

        val = self._find_action_amount(text, self.action_patterns["move_forward"], self.defaults["delta_x"])
        if val is not None:
            result["delta_x"] += abs(val)
            matched.append(f"forward {abs(val)} m")

        val = self._find_action_amount(text, self.action_patterns["move_backward"], self.defaults["delta_x"])
        if val is not None:
            result["delta_x"] -= abs(val)
            matched.append(f"backward {abs(val)} m")

        val = self._find_action_amount(text, self.action_patterns["move_right"], self.defaults["delta_y"])
        if val is not None:
            result["delta_y"] += abs(val)
            matched.append(f"right {abs(val)} m")

        val = self._find_action_amount(text, self.action_patterns["move_left"], self.defaults["delta_y"])
        if val is not None:
            result["delta_y"] -= abs(val)
            matched.append(f"left {abs(val)} m")

        val = self._find_action_amount(text, self.action_patterns["rotate_cw"], self.defaults["delta_yaw"])
        if val is not None:
            result["delta_yaw"] += abs(val)
            matched.append(f"clockwise {abs(val)} deg")

        val = self._find_action_amount(text, self.action_patterns["rotate_ccw"], self.defaults["delta_yaw"])
        if val is not None:
            result["delta_yaw"] -= abs(val)
            matched.append(f"counterclockwise {abs(val)} deg")

        for pat in self.action_patterns["hover"]:
            if re.search(pat, text):
                if not matched:
                    result["action"] = "hover"
                    result["reasoning"] = "Hover command detected."
                    return result

        if not matched:
            return None

        non_zero_axes = sum(
            1 for k in ["delta_x", "delta_y", "delta_z", "delta_yaw"]
            if abs(result[k]) > 1e-9
        )

        if non_zero_axes == 1:
            if result["delta_z"] > 0:
                result["action"] = "ascend"
            elif result["delta_z"] < 0:
                result["action"] = "descend"
            elif result["delta_x"] > 0:
                result["action"] = "move_forward"
            elif result["delta_x"] < 0:
                result["action"] = "move_backward"
            elif result["delta_y"] > 0:
                result["action"] = "move_right"
            elif result["delta_y"] < 0:
                result["action"] = "move_left"
            elif result["delta_yaw"] > 0:
                result["action"] = "rotate_cw"
            elif result["delta_yaw"] < 0:
                result["action"] = "rotate_ccw"
        else:
            result["action"] = "composite"

        result["reasoning"] = "Parsed command deterministically: " + ", ".join(matched) + "."
        return result

    def _apply_defaults(self, cmd: dict) -> dict:
        action_axis_map = {
            "ascend": ("delta_z", +1),
            "descend": ("delta_z", -1),
            "move_forward": ("delta_x", +1),
            "move_backward": ("delta_x", -1),
            "move_right": ("delta_y", +1),
            "move_left": ("delta_y", -1),
            "rotate_cw": ("delta_yaw", +1),
            "rotate_ccw": ("delta_yaw", -1),
        }

        action = cmd.get("action", "")
        if action in action_axis_map:
            axis, sign = action_axis_map[action]
            if float(cmd.get(axis, 0.0)) == 0.0:
                cmd[axis] = sign * self.defaults.get(axis, 5.0)

        if "speed" not in cmd or cmd["speed"] in [None, 0]:
            cmd["speed"] = self.defaults["speed"]

        return cmd

    def parse(self, user_command: str) -> DroneCommand:
        cmd = self._rule_based_parse(user_command)

        if cmd is None:
            raw = self._call_llm(user_command)
            cmd = self._extract_json(raw)

        cmd = self._apply_defaults(cmd)

        return DroneCommand(
            action=cmd.get("action", "hover"),
            delta_x=float(cmd.get("delta_x", 0.0)),
            delta_y=float(cmd.get("delta_y", 0.0)),
            delta_z=float(cmd.get("delta_z", 0.0)),
            delta_yaw=float(cmd.get("delta_yaw", 0.0)),
            speed=float(cmd.get("speed", self.defaults["speed"])),
            reasoning=cmd.get("reasoning", "")
        )


class LLMTaskNode(Node):
    def __init__(self):
        super().__init__("llm_task_node")

        self.parser = LLMCommandParser()

        self.command_sub = self.create_subscription(
            String,
            "/llm/user_command",
            self.command_callback,
            10
        )

        self.parsed_pub = self.create_publisher(
            String,
            "/llm/parsed_command",
            10
        )

        self.get_logger().info("LLM task node started. Listening on /llm/user_command")

    def command_callback(self, msg: String):
        user_text = msg.data.strip()
        self.get_logger().info(f"Received command: {user_text}")

        try:
            result = self.parser.parse(user_text)
            out = String()
            out.data = json.dumps(asdict(result), ensure_ascii=False)
            self.parsed_pub.publish(out)

            self.get_logger().info(f"Published parsed command: {out.data}")

        except Exception as e:
            self.get_logger().error(f"Parsing failed: {e}")


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