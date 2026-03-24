import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import re 
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
import json

try:
    from openai import OpenAI
except ImportError:
    print("LÜTFEN YÜKLE: pip install openai")


try:
    from fusion_msgs.srv import StrategicAdvice
except ImportError:
    pass

USE_LOCAL_LLM = True
LOCAL_MODEL_NAME = "qwen2.5:7b"

class LLMStrategicNode(Node):
    def __init__(self):
        super().__init__('llm_strategic_node')
        
        if USE_LOCAL_LLM:
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            self.get_logger().info(f'🦙 LOKAL LLM (Ollama) Hazır: {LOCAL_MODEL_NAME}')
        else:
            self.client = OpenAI(api_key="sk-...")

        self.latest_system_state = json.dumps({
            "lidar": {"front_space": 0.0}, 
            "status": "WAITING_FOR_SENSORS"
        })

        self.create_subscription(
            String,
            '/llm/system_summary',
            self.state_callback,
            10
        )

        self.srv = self.create_service(
            StrategicAdvice, 
            'llm/plan_and_explain', 
            self.handle_advice_request
        )

        self.get_logger().info("✅ LLM Node Hazır")


    def state_callback(self, msg):
        self.latest_system_state = msg.data

    def handle_advice_request(self, request, response):
        start_t = time.time()
        if request.system_summary_json == "" or request.system_summary_json == "auto":
            summary_str = self.latest_system_state
            source = "REAL SENSORS"
        else:
            summary_str = request.system_summary_json
            source = "MANUAL INPUT"

        mission = request.mission_goal
        constraints = request.constraints
        
        self.get_logger().info(f"📨 LLM Düşünüyor (CoT)... (Kaynak: {source})")

        try:
            llm_result = self.call_llm(summary_str, mission, constraints)
            response.advice_json = json.dumps(llm_result)  
            response.explanation_text = llm_result.get("explanation", "")
            response.success = True
            
        except Exception as e:
            self.get_logger().error(f"LLM İşleme Hatası: {str(e)}")
            response.success = False
            response.explanation_text = "Internal Logic Error"
            response.advice_json = json.dumps({"mode": "cautious", "action": "slow_down", "params": {"speed_limit": 0.3, "collision_weight": 2.0}})

        process_time = time.time() - start_t
        self.get_logger().info(f"✅ Karar ({process_time:.2f}s): {response.explanation_text}")
        return response


    def call_llm(self, context_json, mission, constraints="None"):

        system_prompt = """
You are the Strategic AI Cortex for an autonomous UAV. You MUST fill every field in the response schema without exception.

### OPERATIONAL MANDATE
1. ANALYZE all inputs: speed, lidar, tracked_objects, vehicle status, and mission_goal.
2. EVALUATE risk based on primary_threat score and distance.
3. COMPARE lidar spaces (front, left, right) to find the safest path.
4. VALIDATE mission status and vehicle safety flags (failsafe, gcs).

### STRICT DECISION HIERARCHY
- CRITICAL FAILURE => If vehicle failsafe is true or gcs_connection_lost is true, mode: "recovery", action: "holding".
- RISK > 0.8 OR DIST < 1.0m => mode: "defense", action: "orbit" or "reverse".
- RISK > 0.4 => mode: "cautious", action: "slow_down".
- SPACE ADVANTAGE => If (side_space > front_space + 0.6) AND (side_space > other_side_space), action: "avoid_left/right".
- MISSION LOSS => If path_available is false OR off_path, action: "replan".
- DEFAULT => mode: "normal", action: "maintain".

### RESPONSE REQUIREMENTS (MANDATORY)
Your JSON must include detailed strings for:
- 'risk': Describe the primary threat's class, distance, and exact risk score.
- 'space': Compare exact meter values of front, left, and right.
- 'decision': Explain the logical step-by-step why this action was chosen over others.

### OUTPUT SCHEMA (STRICT JSON ONLY)
{
  "reasoning_trace": {
    "risk": "string (MUST contain threat data)",
    "space": "string (MUST contain lidar comparison)",
    "mission": "string (MUST describe planner status)",
    "decision": "string (Detailed logic)",
    "confidence": float
  },
  "mode": "normal|cautious|defense|holding|recovery",
  "action": "maintain|slow_down|avoid_left|avoid_right|reverse|orbit|replan",
  "params": {
    "speed_limit": float,
    "collision_weight": float,
    "escape_vector": [x, y, 0.0]
  },
  "explanation": "Short summary for operator"
}
"""
       
        user_prompt = f"""
        CURRENT SENSOR DATA: {context_json}
        MISSION GOAL: {mission}
        ACTIVE CONSTRAINTS: {constraints}
        
        EXECUTE COMPREHENSIVE CHAIN-OF-THOUGHT ANALYSIS AND DECIDE.
        MANDATORY VALIDATION CHECKLIST:
        Before finalizing your output, verify:

        ☐ If failsafe is active or gcs_lost is true, did I set mode = "recovery"?
        ☐ If risk_score >= 0.80, did I set mode = "defense"?
        ☐ If risk_score >= 0.75 AND front <= 2.0, did I avoid "maintain"?
        ☐ If one side has +0.6m advantage, did I prefer directional avoidance?
        ☐ If high risk with limited front, did I choose active maneuver (not just slow_down)?
        ☐ Are my speed_limit and collision_weight consistent with mode?

        If ANY checkbox fails, REVISE your decision before outputting.
        No prose before the JSON.
        No prose after the JSON.

        Return strategic decision as strict JSON.
        """

        try:
            response = self.client.chat.completions.create(
                model=LOCAL_MODEL_NAME, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content
            
          
            self.get_logger().info(f"🔍 HAM LLM ÇIKTISI: {content}")
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if match:
                clean_json = match.group(1)
            else:
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    clean_json = content[start:end]
                else:
                    raise ValueError("JSON parantezleri bulunamadı.")
            context = {}
            try:
                context = json.loads(context_json)
            except Exception:
                context = {}

            mission_ctx = context.get("mission", {})
            parsed_result = json.loads(clean_json)
            parsed_result.setdefault("reasoning_trace", {})
            parsed_result.setdefault("params", {})
            trace = parsed_result.get("reasoning_trace", {})
            parsed_result["params"].setdefault(
                "mission_progress",
                mission_ctx.get("mission_progress", 0.0)
            )
            

            parsed_result["reasoning_trace"]["risk_assessment"] = trace.get("risk", "No risk data")
            parsed_result["reasoning_trace"]["space_analysis"] = trace.get("space", "No space data")
            if "mission_analysis" not in parsed_result["reasoning_trace"] or \
            parsed_result["reasoning_trace"]["mission_analysis"] in ["", "Mission analysis not fully provided by model."]:
                
                planner_status = mission_ctx.get("planner_status", "unknown")
                dist_next = mission_ctx.get("distance_to_next_waypoint", None)
                dist_final = mission_ctx.get("distance_to_final_goal", None)
                goal_dir = mission_ctx.get("goal_direction", "unknown")
                prog = mission_ctx.get("mission_progress", None)

                parsed_result["reasoning_trace"]["mission_analysis"] = (
                    f"Planner status: {planner_status}, "
                    f"distance_to_next_waypoint: {dist_next}, "
                    f"distance_to_final_goal: {dist_final}, "
                    f"goal_direction: {goal_dir}, "
                    f"mission_progress: {prog}."
                )
          

           

            if not self.validate_llm_output(parsed_result):
                self.get_logger().warn("⚠️ LLM output validation failed, using fallback")
                raise ValueError("Validation failed")

            return parsed_result
        

        except Exception as e:
            error_msg = f"PARSE ERROR: {str(e)}"
            self.get_logger().error(error_msg)
            
            return {
                    "reasoning_trace": {
                        "risk_assessment": "Parse or validation error occurred",
                        "space_analysis": "Unable to analyze local free space reliably",
                        "mission_analysis": "Mission context could not be interpreted",
                        "decision": "Fallback to cautious safety-preserving behavior",
                        "confidence": 0.3,
                        "alternatives_rejected": {}
                    },
                    "mode": "cautious",
                    "action": "slow_down",
                    "params": {
                        "speed_limit": 0.3,
                        "collision_weight": 2.0,
                        "confidence_score": 0.3,
                        "threat_priority": "none",
                        "escape_vector": [0.0, 0.0, 0.0],
                        "time_to_impact": None,
                        "alternative_actions": [],
                        "mission_progress": 0.0
                    },
                    "explanation": f"LLM parse failed. Fallback to cautious mode. Error: {str(e)}"
                }
        
    def validate_llm_output(self, llm_result):
        try:
            if not isinstance(llm_result, dict):
                self.get_logger().warn("❌ Output is not a dict")
                return False

            required_fields = ["mode", "action", "params"]
            for field in required_fields:
                if field not in llm_result:
                    self.get_logger().warn(f"❌ Missing field: {field}")
                    return False

            valid_modes = [
                "normal",
                "cautious",
                "defense",
                "holding",
                "recovery"
            ]
            if llm_result["mode"] not in valid_modes:
                self.get_logger().warn(f"❌ Invalid mode: {llm_result['mode']}")
                return False

            valid_actions = [
                "maintain",
                "slow_down",
                "avoid_left",
                "avoid_right",
                "reverse",
                "orbit",
                "replan"
            ]
            if llm_result["action"] not in valid_actions:
                self.get_logger().warn(f"❌ Invalid action: {llm_result['action']}")
                return False

            if not isinstance(llm_result["params"], dict):
                self.get_logger().warn("❌ params is not a dict")
                return False

            return True

        except Exception as e:
            self.get_logger().error(f"❌ Validation error: {str(e)}")
            return False

def main(args=None):
    rclpy.init(args=args)
    node = LLMStrategicNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()