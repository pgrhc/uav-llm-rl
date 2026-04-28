#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import threading
from typing import Dict, List, Optional, Tuple

import requests
import rclpy
from pydantic import ValidationError
from rclpy.node import Node
from std_msgs.msg import String

from uav_llm.mission_models import (
    IntentType,
    InterpretedCommand,
    MissionContext,
)


# =========================
# NORMALIZATION
# =========================

INTENT_ALIAS_MAP: Dict[str, str] = {
    "hide":                 "hide_or_relocate",
    "hide_or_seek_cover":   "hide_or_relocate",
    "seek_cover":           "hide_or_relocate",
    "conceal":              "hide_or_relocate",
    "relocate_to_cover":    "hide_or_relocate",
}

EXPLAINABILITY_BOILERPLATE = {
    "this step is safe",
    "no issues expected",
    "derived from user request",
    "generic planning policy",
    "this step may rely on heuristic",
    "most relevant available target",
    "considered safe under current planning assumptions",
    "no additional environmental evidence was provided",
}


def _is_boilerplate(text: str) -> bool:
    return text.lower().strip() in EXPLAINABILITY_BOILERPLATE


def normalize_intent(parsed: dict) -> Tuple[dict, List[str]]:
    changes: List[str] = []
    interpreted = parsed.get("interpreted_command", {}) or {}
    raw_intent = interpreted.get("intent")
    if isinstance(raw_intent, str) and raw_intent in INTENT_ALIAS_MAP:
        corrected = INTENT_ALIAS_MAP[raw_intent]
        interpreted["intent"] = corrected
        changes.append(f"intent '{raw_intent}' -> '{corrected}'")
    return parsed, changes


def repair_intent_fields(
    parsed: dict, context: Optional[MissionContext] = None
) -> dict:
    """
    patrol_or_explore ve hide_or_relocate için
    target_area ve location_hint boşsa context'ten doldur.
    """
    try:
        interpreted = parsed.get("interpreted_command", {}) or {}
        location_hint = interpreted.get("location_hint") or {}
        intent = interpreted.get("intent")

        floor = location_hint.get("floor")
        if floor is None and context is not None:
            floor = context.current_floor

        target_area = interpreted.get("target_area")
        if not target_area and context is not None and context.current_area:
            target_area = context.current_area
            interpreted["target_area"] = target_area

        if intent == "hide_or_relocate" and not target_area:
            interpreted["target_area"] = "nearest_concealment_zone"

        if (
            intent == "patrol_or_explore"
            and not target_area
            and floor is None
            and context is not None
            and context.current_floor is not None
        ):
            interpreted["target_area"] = "current_floor"
            interpreted.setdefault("location_hint", {})
            interpreted["location_hint"]["floor"] = context.current_floor

    except Exception:
        pass
    return parsed


# =========================
# PROMPT
# =========================

EXPLAINABILITY_EXAMPLES = """
=== EXPLAINABILITY FIELD QUALITY EXAMPLES ===

BAD (will be rejected):
  justification: "User wants to find a book."
  success_criteria: ["book found"]

GOOD (accepted):
  justification: "User issued a search_and_find command targeting a red-covered book
    on floor 3. Priority is normal. No constraints mentioned."
  success_criteria: [
    "red-covered book located and position reported",
    "all rooms on floor 3 searched or marked unreachable"
  ]
"""

INTENT_RULES = """
=== INTENT RULES ===
- search_and_find: locating a specific object or entity
- inspect_area: focused inspection of ONE specific area
- track_target: following or maintaining contact with a moving entity
- verify_event: confirming or denying a reported event or observation
- deliver_or_report: generating and sending a report or result
- patrol_or_explore: multi-area coverage, "check all rooms", "scan this floor"
- priority_override: temporarily doing something else before resuming
- cancel_mission: stopping the whole mission
- cancel_step: stopping only the current subtask
- hide_or_relocate: moving to cover, concealment, or a safe position
- unknown: use only if intent cannot be determined at all

Do NOT use action names as intent values.
"""


def build_interpreter_prompt(
    envelope: dict,
    context: Optional[MissionContext] = None,
    validation_feedback: Optional[str] = None,
) -> str:
    ctx_block = ""
    if context is not None:
        ctx = context.as_dict()
        if any(v is not None for v in ctx.values()):
            ctx_block = (
                "\nCurrent navigation context "
                "(use when user says 'this floor', 'here', 'current room'):\n"
                + json.dumps(ctx, ensure_ascii=False, indent=2)
                + "\n"
            )

    extra = ""
    if validation_feedback:
        extra = (
            "\n=== PREVIOUS ATTEMPT FAILED VALIDATION ===\n"
            "Return ONLY corrected JSON.\n"
            f"Error:\n{validation_feedback}\n"
        )

    user_payload = {
        "command_id": envelope["command_id"],
        "mission_id": envelope["mission_id"],
        "timestamp":  envelope["timestamp"],
        "language":   envelope.get("language", "auto"),
        "raw_text":   envelope["raw_text"],
    }

    return f"""
You are a mission interpreter for an autonomous robotic system.

Your ONLY job is to interpret the user command into a normalized InterpretedCommand.
Do NOT produce a task plan. Do NOT produce steps.

Return ONLY valid JSON.
Do NOT return markdown. Do NOT return code fences.

{INTENT_RULES}
{EXPLAINABILITY_EXAMPLES}
{ctx_block}

=== SUPPORTED INTENTS ===
search_and_find | inspect_area | track_target | verify_event |
deliver_or_report | patrol_or_explore | priority_override |
cancel_mission | cancel_step | hide_or_relocate | unknown

=== OUTPUT SCHEMA ===
{{
  "command_id": "string",
  "mission_id": "string",
  "timestamp": "string",
  "intent": "one of the supported intents",
  "priority": "low|normal|high|urgent",
  "source": "user_command|user_live_observation|operator_override|system_generated",
  "target": {{"type": "string", "attributes": ["string"]}} | null,
  "target_area": "string | null",
  "target_entity": {{"type": "string", "attributes": ["string"]}} | null,
  "location_hint": {{
    "floor": 0|null,
    "room": "string|null",
    "corridor": "string|null",
    "area": "string|null"
  }} | null,
  "constraints": ["string"],
  "success_criteria": ["string"],
  "secondary_objectives": ["string"],
  "execution_modifiers": ["string"],
  "justification": "string — explain how you interpreted this command and why",
  "raw_text": "string — copy the original user text here"
}}

RULES:
- justification must be specific: why this intent, what evidence from the command.
- success_criteria must be concrete and measurable.
- secondary_objectives: list any implicit sub-goals (e.g. "safe_land", "report_result").
- execution_modifiers: list any conditions like "pause_current_plan", "resume_after".
- For any list field return [] instead of null.
- Do not invent precise details not present in the command.
{extra}

Input:
{json.dumps(user_payload, ensure_ascii=False, indent=2)}
"""


# =========================
# ROS NODE
# =========================

class MissionInterpreterNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_interpreter_node")

        self.declare_parameter("ollama_url", "http://localhost:11434/api/generate")
        self.declare_parameter("model_name", "qwen2.5:7b")
        self.declare_parameter("request_timeout_sec", 120.0)

        self.ollama_url = (
            self.get_parameter("ollama_url").get_parameter_value().string_value
        )
        self.model_name = (
            self.get_parameter("model_name").get_parameter_value().string_value
        )
        self.request_timeout_sec = (
            self.get_parameter("request_timeout_sec").get_parameter_value().double_value
        )

        self.mission_context = MissionContext()

        # stats
        self._stats = {"total": 0, "retry": 0, "failed": 0}
        self.create_timer(60.0, self._log_stats)

        self.sub = self.create_subscription(
            String,
            "/mission/user_command/envelope",
            self._envelope_callback,
            10,
        )
        self.output_pub = self.create_publisher(
            String, "/mission/interpreted_command", 10
        )
        self.error_pub = self.create_publisher(
            String, "/mission/interpreter/error", 10
        )

        self.get_logger().info("mission_interpreter_node started.")
        self.get_logger().info(f"model: {self.model_name}")
        self.get_logger().info(f"ollama: {self.ollama_url}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish_json(self, pub, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)

    def _extract_json_object(self, text: str) -> str:
        """Bracket sayacı — greedy regex yerine dengeli JSON bulur."""
        depth, start = 0, None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start: i + 1]
        raise ValueError("No balanced JSON object found in model response")

    def _call_llm(self, prompt: str) -> str:
        payload = {
            "model":   self.model_name,
            "prompt":  prompt,
            "stream":  False,
            "format":  "json",
            "options": {"temperature": 0.2},
        }
        response = requests.post(
            self.ollama_url, json=payload, timeout=self.request_timeout_sec
        )
        response.raise_for_status()
        data = response.json()
        if "response" not in data:
            raise ValueError("LLM response missing 'response' field")
        return data["response"]

    def _publish_error(
        self,
        command_id: str,
        mission_id: str,
        error_type: str,
        error_message: str,
        raw_response: Optional[str] = None,
    ) -> None:
        self._publish_json(
            self.error_pub,
            {
                "command_id":    command_id,
                "mission_id":    mission_id,
                "error_type":    error_type,
                "error_message": error_message,
                "raw_response":  raw_response,
            },
        )

    def _validate(
        self, parsed: dict, context: Optional[MissionContext], logger=None
    ) -> InterpretedCommand:
        """normalize → repair → Pydantic."""
        parsed, intent_changes = normalize_intent(parsed)
        if logger and intent_changes:
            for c in intent_changes:
                logger.warn(f"[IntentNorm] {c}")

        parsed = repair_intent_fields(parsed, context=context)
        return InterpretedCommand(**parsed)

    def _check_justification_quality(self, interpreted: InterpretedCommand) -> None:
        """Boilerplate justification'ı warn ile logla."""
        if _is_boilerplate(interpreted.justification):
            self.get_logger().warn(
                f"[Quality] justification boilerplate detected "
                f"for command_id={interpreted.command_id}"
            )

    def _log_stats(self) -> None:
        t = self._stats["total"]
        r = self._stats["retry"]
        f = self._stats["failed"]
        self.get_logger().info(
            f"[Stats] total={t} retry={r} failed={f} "
            f"retry_rate={r / max(t, 1):.1%} "
            f"fail_rate={f / max(t, 1):.1%}"
        )

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _envelope_callback(self, msg: String) -> None:
        # envelope parse
        raw_envelope: dict = {}
        try:
            raw_envelope = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid envelope JSON: {e}")
            return

        command_id = raw_envelope.get("command_id", "unknown")
        mission_id = raw_envelope.get("mission_id", "unknown")
        self.get_logger().info(
            f"Interpret request | command_id={command_id}"
        )

        prompt = build_interpreter_prompt(
            raw_envelope, context=self.mission_context
        )

        # LLM call
        try:
            raw_llm = self._call_llm(prompt)
        except Exception as e:
            self.get_logger().error(f"LLM call failed: {e}")
            self._publish_error(command_id, mission_id, "llm_call_failed", str(e))
            self._stats["failed"] += 1
            return

        # JSON parse
        parsed: Optional[dict] = None
        try:
            parsed = json.loads(self._extract_json_object(raw_llm))
        except Exception as e:
            self.get_logger().error(f"JSON parse failed: {e}")
            self._publish_error(
                command_id, mission_id, "json_parse_failed", str(e),
                raw_response=raw_llm,
            )
            self._stats["failed"] += 1
            return

        # Validation — tier 1
        try:
            interpreted = self._validate(
                parsed, self.mission_context, logger=self.get_logger()
            )
            self._stats["total"] += 1

        except ValidationError as e_first:
            self.get_logger().warn(
                f"Validation failed, retrying | command_id={command_id} | {e_first}"
            )
            self._stats["retry"] += 1

            # Tier 2
            retry_parsed: Optional[dict] = None
            retry_raw: Optional[str] = None
            try:
                retry_prompt = build_interpreter_prompt(
                    raw_envelope,
                    context=self.mission_context,
                    validation_feedback=str(e_first),
                )
                retry_raw = self._call_llm(retry_prompt)
                retry_parsed = json.loads(self._extract_json_object(retry_raw))
                interpreted = self._validate(
                    retry_parsed, self.mission_context, logger=self.get_logger()
                )
                self._stats["total"] += 1

            except ValidationError as e_second:
                self.get_logger().error(
                    f"Validation failed after retry | command_id={command_id} | {e_second}"
                )
                self._publish_error(
                    command_id, mission_id,
                    "schema_validation_failed", str(e_second),
                    raw_response=json.dumps(
                        retry_parsed or parsed, ensure_ascii=False
                    ),
                )
                self._stats["failed"] += 1
                return

            except Exception as e_retry:
                self.get_logger().error(f"Retry failed: {e_retry}")
                self._publish_error(
                    command_id, mission_id,
                    "retry_failed", str(e_retry),
                    raw_response=retry_raw or raw_llm,
                )
                self._stats["failed"] += 1
                return

        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")
            self._publish_error(
                command_id, mission_id,
                "unexpected_error", str(e),
                raw_response=json.dumps(parsed or {}, ensure_ascii=False),
            )
            self._stats["failed"] += 1
            return

        self._check_justification_quality(interpreted)

        self._publish_json(self.output_pub, interpreted.model_dump())
        self.get_logger().info(
            f"Interpreted | command_id={interpreted.command_id} | "
            f"intent={interpreted.intent.value} | "
            f"priority={interpreted.priority.value}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionInterpreterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down mission_interpreter_node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()