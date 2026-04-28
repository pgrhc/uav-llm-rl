#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import requests
import rclpy
from pydantic import ValidationError
from rclpy.node import Node
from std_msgs.msg import String

from uav_llm.mission_models import (
    ActionType,
    InterpretedCommand,
    TaskPlan,
)


# =========================
# NORMALIZATION
# =========================

ACTION_ALIAS_MAP: Dict[str, str] = {
    "land_on_vehicle":       "safe_land",
    "land_on_target":        "safe_land",
    "approach_and_land":     "safe_land",
    "dock":                  "safe_land",
    "docking":               "safe_land",
    "perch":                 "safe_land",
    "approach_vehicle":      "approach_target",
    "hide":                  "hide_or_seek_cover",
    "seek_cover":            "hide_or_seek_cover",
    "conceal":               "hide_or_seek_cover",
    "resume_plan":           "resume_previous_plan",
    "return_to_current_plan": "resume_previous_plan",
    "continue_previous_plan": "resume_previous_plan",
    "pause_plan":            "pause_current_plan",
    "hold_current_plan":     "pause_current_plan",
    "rtb":                   "return_to_base",
    "go_home":               "return_to_base",
    "sweep_area":            "search_area",
    "clear_area":            "search_area",
    "scan_area":             "search_area",
    "scan_room":             "search_room",
    "check_room":            "search_room",
    "clear_room":            "search_room",
    "deliver_report":        "report_result",
    "send_report":           "report_result",
    "broadcast_result":      "report_result",
    "hover":                 "wait_and_observe",
    "loiter":                "wait_and_observe",
    "standby":               "wait_and_observe",
    "hold_position":         "wait_and_observe",
    "terminate_mission":     "cancel_active_mission",
    "abort_mission":         "cancel_active_mission",
    "stop_mission":          "cancel_active_mission",
    "terminate_step":        "cancel_active_step",
    "abort_step":            "cancel_active_step",
    "complete_mission":      "complete_mission",
    "finalize_mission":      "complete_mission",
    "end_mission":           "complete_mission",
    "mission_complete":      "complete_mission",
    "finish_task":           "complete_mission",
    "close_mission":         "complete_mission",
    "patrol":                "search_area",
}

VALID_ACTIONS = set(a.value for a in ActionType)

# Boilerplate kalıpları — bunlar Pydantic'e gitmeden reddedilir
BOILERPLATE_PHRASES = {
    "this step is safe",
    "this step is considered safe",
    "no issues expected",
    "derived from user request",
    "generic planning policy",
    "most relevant available target",
    "this step may rely on heuristic",
    "considered safe under current planning assumptions",
    "no additional environmental evidence was provided",
    "should be re-evaluated during execution",
}

# Minimum karakter uzunlukları
MIN_TARGET_SELECTION_REASON = 40
MIN_SAFETY_RATIONALE = 40
MIN_DECISION_BASIS_ITEMS = 1
MIN_DECISION_BASIS_ITEM_LENGTH = 15  # "user_command" tek başına yetmez


# =========================
# HELPERS
# =========================

def _is_boilerplate(text: str) -> bool:
    t = text.lower().strip()
    return t in BOILERPLATE_PHRASES or len(t) < 10


def normalize_actions(parsed: dict) -> Tuple[dict, List[str]]:
    changes: List[str] = []
    for step in parsed.get("steps", []):
        raw_action = step.get("action")
        if not isinstance(raw_action, str):
            continue
        if raw_action in VALID_ACTIONS:
            continue
        if raw_action in ACTION_ALIAS_MAP:
            corrected = ACTION_ALIAS_MAP[raw_action]
            changes.append(
                f"step {step.get('step_id')}: '{raw_action}' -> '{corrected}'"
            )
            step["action"] = corrected
            if raw_action in {
                "land_on_vehicle", "land_on_target",
                "approach_and_land", "dock", "docking", "perch",
            }:
                step["reason"] = (
                    "[normalized landing action] " + step.get("reason", "")
                )
    return parsed, changes


def repair_list_fields(parsed: dict) -> dict:
    """Sadece liste alanlarını onar — explainability alanlarına dokunma."""
    for step in parsed.get("steps", []):
        for field in ("decision_basis", "assumptions", "uncertainty_notes",
                      "evidence", "preconditions"):
            if step.get(field) is None:
                step[field] = []
    return parsed


def check_explainability(parsed: dict) -> List[str]:
    """
    Boilerplate ve minimum kalite kontrolü.
    Sorunlu step_id + alan adlarını döner.
    Pydantic'e gitmeden önce çalışır — LLM'i tekrar zorlar.
    """
    issues: List[str] = []

    for step in parsed.get("steps", []):
        sid = step.get("step_id", "?")

        tsr = step.get("target_selection_reason", "")
        if not tsr or len(tsr) < MIN_TARGET_SELECTION_REASON or _is_boilerplate(tsr):
            issues.append(
                f"step {sid} target_selection_reason: too vague or boilerplate "
                f"(min {MIN_TARGET_SELECTION_REASON} chars, specific context required)"
            )

        sr = step.get("safety_rationale", "")
        if not sr or len(sr) < MIN_SAFETY_RATIONALE or _is_boilerplate(sr):
            issues.append(
                f"step {sid} safety_rationale: too vague or boilerplate "
                f"(min {MIN_SAFETY_RATIONALE} chars, specific safety reasoning required)"
            )

        db = step.get("decision_basis") or []
        if len(db) < MIN_DECISION_BASIS_ITEMS:
            issues.append(
                f"step {sid} decision_basis: must have at least "
                f"{MIN_DECISION_BASIS_ITEMS} item(s)"
            )
        elif all(len(item) < MIN_DECISION_BASIS_ITEM_LENGTH for item in db):
            issues.append(
                f"step {sid} decision_basis: items too vague — "
                "include specific references to the command, observation, or constraint"
            )

    return issues


def normalize_plan_output(
    parsed: dict, logger=None
) -> Tuple[dict, List[str]]:
    """
    Pipeline:
      1. action alias normalizasyonu
      2. liste alanları onarımı
      3. explainability kalite kontrolü
    Returns: (parsed, quality_issues)
    """
    parsed, action_changes = normalize_actions(parsed)
    if logger and action_changes:
        for c in action_changes:
            logger.warn(f"[ActionNorm] {c}")

    parsed = repair_list_fields(parsed)

    quality_issues = check_explainability(parsed)

    return parsed, quality_issues



EXPLAINABILITY_EXAMPLES = """
Good explainability fields must reference command, scene memory, target choice, safety condition, assumptions, uncertainty, and evidence. Avoid generic phrases.
"""
def compact_scene_summary(scene):
    if scene is None:
        return None

    sm = scene.get("scene_memory", {})
    th = scene.get("dynamic_threats", {})
    hint = scene.get("planner_hint", {})

    return {
        "confirmed_objects": [
            {
                "label": o.get("label"),
                "confidence": o.get("confidence"),
                "position_2d": o.get("position_2d"),
                "range_m": o.get("range_m"),
                "direction": o.get("direction"),
                "state": o.get("state"),
            }
            for o in sm.get("confirmed_objects", [])[:5]
        ],
        "nearest_object": sm.get("nearest_confirmed_object"),
        "lost_objects": [
            {
                "label": o.get("label"),
                "position_2d": o.get("position_2d"),
                "direction": o.get("direction"),
            }
            for o in sm.get("lost_objects", [])[:2]
        ],
        "risk_level": th.get("risk_level", "unknown"),
        "max_risk": th.get("max_risk", 0.0),
        "planner_hint": hint.get("text_hints", [])[:3],
    }

def analyze_scene_for_target(target_label: Optional[str], scene: Optional[dict]) -> dict:
    """
    Python karar vermez — sadece sahneyi target'a göre analiz edip
    LLM'in görmesini kolaylaştıracak şekilde yapılandırır.
    """
    if not scene or not target_label:
        return {"target_analysis_available": False}

    target_lower = target_label.lower()
    confirmed = scene.get("confirmed_objects", [])
    lost = scene.get("lost_objects", [])

    matches_confirmed = [
        obj for obj in confirmed
        if target_lower in obj.get("label", "").lower()
        or obj.get("label", "").lower() in target_lower
    ]

    matches_lost = [
        obj for obj in lost
        if target_lower in obj.get("label", "").lower()
        or obj.get("label", "").lower() in target_lower
    ]

    return {
        "target_analysis_available": True,
        "searched_for": target_label,
        "found_in_confirmed": len(matches_confirmed) > 0,
        "found_in_lost": len(matches_lost) > 0,
        "confirmed_matches": matches_confirmed,   # LLM bunları görür, kararı LLM verir
        "lost_matches": matches_lost,
    }



def build_planner_prompt(interpreted, scene_summary=None, target_analysis=None, validation_feedback=None, quality_issues=None):
    extra = ""
    if validation_feedback:
        extra += f"\nFIX VALIDATION ERROR:\n{validation_feedback}\n"
    if quality_issues:
        extra += "\nFIX QUALITY ISSUES:\n" + "\n".join(f"- {q}" for q in quality_issues) + "\n"
    analysis_block = ""
    if target_analysis and target_analysis.get("target_analysis_available"):
        analysis_block = (
            "\nTARGET ANALYSIS (pre-computed from scene):\n"
            + json.dumps(target_analysis, ensure_ascii=False)
            + "\n"
        )
    scene_block = ""
    if scene_summary is not None:
        scene_block = (
            "\nSCENE:\n"
            "confirmed_objects = reliable\n"
            "candidate_objects = uncertain\n"
            "lost_objects = for reacquire only\n"
            "ghost_objects = ignore\n"
            + json.dumps(scene_summary, ensure_ascii=False)
            + "\n"
        )
    


    return f"""You are a robot task planner. Return ONLY valid JSON.

ACTIONS:
reach_floor, navigate_to_area, search_area, search_room, inspect_area,
inspect_object_candidate, track_entity, verify_event, report_result, wait_and_observe,
reacquire_target, abort_and_replan, enumerate_searchable_areas, mark_area_searched,
cancel_active_mission, cancel_active_step, approach_target, safe_land, complete_mission,
resume_previous_plan, pause_current_plan, hide_or_seek_cover, return_to_base

RULES:
- Use only ACTIONS.
- confirmed object matches target -> approach_target, inspect_object_candidate, report_result.
- For known objects use approach_target, not navigate_to_area.
- For areas/floors/rooms/corridors use navigate_to_area.
- Last step must be report_result or complete_mission.
- Target-required actions must have non-null string target.
- evidence must contain real scene values, not field names.
- fallback_if_failed must never be null. Default: abort_and_replan.
- Ignore ghost objects.
- 2-5 steps max.
STRICT RULE:
- If any confirmed object label contains the target word or the target word contains the label, use that confirmed object; e.g. target "table" matches "dining_table". Do not use nearest_object unless no target match exists.
- If intent is hide_or_relocate and target_entity exists, after approach/inspect add hide_or_seek_cover with target=target_entity.
TARGET REQUIRED ACTIONS:
approach_target, inspect_object_candidate, navigate_to_area, search_area,
search_room, inspect_area, verify_event, safe_land, hide_or_seek_cover,
return_to_base

SCHEMA:
{{
  "command_id": "{interpreted.get('command_id')}",
  "mission_id": "{interpreted.get('mission_id')}",
  "timestamp": "ISO8601",
  "plan_reasoning": "string",
  "replannable": true,
  "steps": [{{
    "step_id": "step_01",
    "action": "ACTIONS value",
    "target": "string or null",
    "reason": "string",
    "target_selection_reason": "string",
    "decision_basis": ["string"],
    "assumptions": ["string"],
    "uncertainty_notes": ["string"],
    "evidence": ["string"],
    "safety_rationale": "string",
    "expected_outcome": "string",
    "success_condition": "string",
    "preconditions": ["string"],
    "fallback_if_failed": "abort_and_replan",
    "interruptible": true,
    "priority": "low|normal|high|urgent",
    "status": "pending"
  }}]
}}

{extra}
{analysis_block}
{scene_block}

COMMAND:
{json.dumps(interpreted, ensure_ascii=False)}
"""


class TaskPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("task_planner_node")

        self.declare_parameter("ollama_url", "http://localhost:11434/api/generate")
        self.declare_parameter("model_name", "qwen2.5:7b")
        self.declare_parameter("request_timeout_sec", 120.0)
        self.latest_scene_summary = None

        self.scene_sub = self.create_subscription(
            String,
            "/llm/scene_summary",
            self._scene_summary_callback,
            10,
        )

        self.ollama_url = (
            self.get_parameter("ollama_url").get_parameter_value().string_value
        )
        self.model_name = (
            self.get_parameter("model_name").get_parameter_value().string_value
        )
        self.request_timeout_sec = (
            self.get_parameter("request_timeout_sec").get_parameter_value().double_value
        )

        # stats
        self._stats = {"total": 0, "retry": 0, "failed": 0, "quality_retry": 0}
        self.create_timer(60.0, self._log_stats)

        self.sub = self.create_subscription(
            String,
            "/mission/interpreted_command",
            self._interpreted_callback,
            10,
        )
        self.output_pub = self.create_publisher(
            String, "/mission/planning/output", 10
        )
        self.error_pub = self.create_publisher(
            String, "/mission/planner/error", 10
        )

        self.get_logger().info("task_planner_node started.")
        self.get_logger().info(f"model: {self.model_name}")
        self.get_logger().info(f"ollama: {self.ollama_url}")


    def _publish_json(self, pub, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)


    def _scene_summary_callback(self, msg: String) -> None:
        try:
            self.latest_scene_summary = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"scene_summary parse failed: {e}")


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

    def _log_stats(self) -> None:
        t = self._stats["total"]
        r = self._stats["retry"]
        f = self._stats["failed"]
        q = self._stats["quality_retry"]
        self.get_logger().info(
            f"[Stats] total={t} retry={r} quality_retry={q} failed={f} "
            f"retry_rate={r / max(t, 1):.1%} "
            f"fail_rate={f / max(t, 1):.1%}"
        )

    def _call_and_parse(self, prompt: str) -> dict:
        """LLM çağır → JSON parse → dict döndür."""
        raw = self._call_llm(prompt)
        return json.loads(self._extract_json_object(raw))

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _interpreted_callback(self, msg: String) -> None:
        # InterpretedCommand parse
        command_id = "unknown"
        mission_id = "unknown"
        interpreted: Optional[InterpretedCommand] = None
        raw_parsed: dict = {}

        try:
            raw_parsed = json.loads(msg.data)
            command_id = raw_parsed.get("command_id", "unknown")
            mission_id = raw_parsed.get("mission_id", "unknown")
            interpreted = InterpretedCommand(**raw_parsed)
        except Exception as e:
            self.get_logger().error(
                f"Invalid InterpretedCommand | command_id={command_id} | {e}"
            )
            self._publish_error(
                command_id, mission_id,
                "invalid_interpreted_command", str(e),
            )
            self._stats["failed"] += 1
            return

        self.get_logger().info(
            f"Plan request | command_id={command_id} | intent={interpreted.intent.value}"
        )

        # raw_text planlayıcıya gerekmiyor — token israfını önle
        interpreted_dict = interpreted.model_dump(exclude={"raw_text"})

        # ── Tier 1 ────────────────────────────────────────────────────
        plan_parsed: Optional[dict] = None
        quality_issues: List[str] = []
        scene_for_prompt = compact_scene_summary(self.latest_scene_summary)
        target_label = None

        if interpreted.target is not None:
            target_label = getattr(interpreted.target, "type", None)

        if not target_label and interpreted.target_entity is not None:
            target_label = getattr(interpreted.target_entity, "type", None)

        target_analysis = analyze_scene_for_target(target_label, scene_for_prompt)

        try:
            

            prompt = build_planner_prompt(
                interpreted_dict,
                scene_summary=scene_for_prompt,
                target_analysis=target_analysis
            )
            plan_parsed = self._call_and_parse(prompt)
            plan_parsed, quality_issues = normalize_plan_output(
                plan_parsed, logger=self.get_logger()
            )

            # Explainability kalite sorunu varsa retry
            if quality_issues:
                self._stats["quality_retry"] += 1
                self.get_logger().warn(
                    f"[Quality] {len(quality_issues)} explainability issue(s), "
                    f"retrying | command_id={command_id}"
                )
                raise ValidationError.from_exception_data(  # Tier 2'ye düş
                    title="ExplainabilityCheck",
                    input_type="python",
                    line_errors=[],
                ) if False else _ExplainabilityError(quality_issues)

            plan = TaskPlan(**plan_parsed)
            self._stats["total"] += 1

        except (_ExplainabilityError, ValidationError) as e_first:
            feedback = (
                "\n".join(e_first.issues)
                if isinstance(e_first, _ExplainabilityError)
                else str(e_first)
            )
            self.get_logger().warn(
                f"Tier 1 failed, retrying | command_id={command_id} | {feedback[:200]}"
            )
            if not isinstance(e_first, _ExplainabilityError):
                self._stats["retry"] += 1

            # ── Tier 2 ────────────────────────────────────────────────
            retry_parsed: Optional[dict] = None
            retry_raw: Optional[str] = None

            try:
                retry_prompt = build_planner_prompt(
                    interpreted_dict,
                    scene_summary=scene_for_prompt,
                    target_analysis=target_analysis,
                    validation_feedback=(
                        feedback
                        if isinstance(e_first, _ExplainabilityError)
                        else str(e_first)
                    ),
                    quality_issues=(
                        e_first.issues
                        if isinstance(e_first, _ExplainabilityError)
                        else quality_issues or None
                    ),
                )
                retry_parsed = self._call_and_parse(retry_prompt)
                retry_parsed, retry_quality = normalize_plan_output(
                    retry_parsed, logger=self.get_logger()
                )
                if retry_quality:
                    self.get_logger().warn(
                        f"[Quality] Retry still has {len(retry_quality)} issue(s) "
                        f"— publishing anyway | command_id={command_id}"
                    )

                plan = TaskPlan(**retry_parsed)
                self._stats["total"] += 1

            except ValidationError as e_second:
                self.get_logger().error(
                    f"Validation failed after retry | command_id={command_id} | {e_second}"
                )
                self._publish_error(
                    command_id, mission_id,
                    "planner_validation_failed", str(e_second),
                    raw_response=json.dumps(
                        retry_parsed or plan_parsed or {}, ensure_ascii=False
                    ),
                )
                self._stats["failed"] += 1
                return

            except Exception as e_retry:
                self.get_logger().error(
                    f"Retry failed | command_id={command_id} | {e_retry}"
                )
                self._publish_error(
                    command_id, mission_id,
                    "planner_retry_failed", str(e_retry),
                )
                self._stats["failed"] += 1
                return

        except Exception as e:
            self.get_logger().error(
                f"Unexpected error | command_id={command_id} | {e}"
            )
            self._publish_error(
                command_id, mission_id,
                "planner_failed", str(e),
                raw_response=json.dumps(plan_parsed or {}, ensure_ascii=False),
            )
            self._stats["failed"] += 1
            return

        self._publish_json(self.output_pub, plan.model_dump())
        self.get_logger().info(
            f"Plan published | "
            f"command_id={plan.command_id} | "
            f"mission_id={plan.mission_id} | "
            f"intent={interpreted.intent.value} | "
            f"steps={len(plan.steps)}"
        )


# Explainability kalite hataları için hafif exception sınıfı
class _ExplainabilityError(Exception):
    def __init__(self, issues: List[str]):
        self.issues = issues
        super().__init__("\n".join(issues))


# =========================
# MAIN
# =========================

def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down task_planner_node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()