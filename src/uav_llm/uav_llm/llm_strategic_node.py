import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import re # Regex için gerekli

# OpenAI / Ollama Kütüphanesi
try:
    from openai import OpenAI
except ImportError:
    print("LÜTFEN YÜKLE: pip install openai")

# Servis Dosyası
try:
    from fusion_msgs.srv import StrategicAdvice
except ImportError:
    pass

# --- AYARLAR ---
USE_LOCAL_LLM = True
LOCAL_MODEL_NAME = "llama3" # "qwen2.5:3b" veya "mistral" de olabilir

class LLMStrategicNode(Node):
    def __init__(self):
        super().__init__('llm_strategic_node')
        
        # 1. LLM İSTEMCİSİ (Ollama)
        if USE_LOCAL_LLM:
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            self.get_logger().info(f'🦙 LOKAL LLM (Ollama) Hazır: {LOCAL_MODEL_NAME}')
        else:
            self.client = OpenAI(api_key="sk-...")

        # 2. HAFIZA (Sensör verisi burada tutulur)
        self.latest_system_state = json.dumps({
            "lidar": {"front_space": 0.0}, 
            "status": "WAITING_FOR_SENSORS"
        })

        # 3. KULAK (Subscriber): StateSummarizer'ı dinler
        self.create_subscription(
            String,
            '/llm/system_summary',
            self.state_callback,
            10
        )

        # 4. SERVİS (Server): Emir bekler
        self.srv = self.create_service(
            StrategicAdvice, 
            'llm/plan_and_explain', 
            self.handle_advice_request
        )

        self.get_logger().info("✅ LLM Node Hazır: Gelişmiş CoT Prompt ile çalışıyor...")

    # --- SENSÖR VERİSİNİ GÜNCELLEME ---
    def state_callback(self, msg):
        """Sensörlerden gelen veriyi hafızaya yazar."""
        self.latest_system_state = msg.data

    # --- SERVİS ÇAĞRISI GELDİĞİNDE ---
    def handle_advice_request(self, request, response):
        start_t = time.time()
        
        # "auto" modundaysa sensör verisini kullan, değilse elle girileni kullan
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
            # Sizin özel Prompt fonksiyonunuzu çağırıyoruz
            llm_result = self.call_llm(summary_str, mission, constraints)
            
            # Cevabı ROS servisine paketle
            response.advice_json = json.dumps({
                "mode": llm_result.get("mode", "normal"),
                "action": llm_result.get("action", "maintain"),
                "params": llm_result.get("params", {}),
                "reasoning_trace": llm_result.get("reasoning_trace", "") # CoT izini de ekleyebiliriz isterseniz
            })
            response.explanation_text = llm_result.get("explanation", "No explanation.")
            response.success = True
            
        except Exception as e:
            self.get_logger().error(f"LLM İşleme Hatası: {str(e)}")
            response.success = False
            response.explanation_text = "Internal Logic Error"
            response.advice_json = json.dumps({"mode": "cautious", "action": "hover"})

        process_time = time.time() - start_t
        self.get_logger().info(f"✅ Karar ({process_time:.2f}s): {response.explanation_text}")
        return response

    # --- SİZİN İSTEDİĞİNİZ ÖZEL PROMPT FONKSİYONU ---
    def call_llm(self, context_json, mission, constraints="None"):
        """
        Ollama Modelini Çağırır - CoT (Chain-of-Thought) Destekli
        """
        
        system_prompt = """
        You are the Strategic AI Cortex for an autonomous UAV, employing Chain-of-Thought (CoT) reasoning.
        
        INPUT DATA:
        - Sensor Data (JSON): Lidar distances, Radar objects, Threat Agent scores.
        - Mission Goal: The objective to achieve.
        - Constraints: Safety limits (e.g., max speed, no-fly zones).

        TASK:
        1. ANALYZE: Review 'primary_threat' scores and 'lidar' spatial data step-by-step.
        2. REASON: Evaluate safety risks against the mission goal. Consider constraints.
        3. DECIDE: Select the optimal 'mode' and 'action'.

        Output: STRICT JSON object only. NO Markdown codes (```), NO intro text.
        
        Response Schema:
        {
            "reasoning_trace": "1. Threat analysis: [Score X detected] -> High/Low risk. 2. Spatial analysis: [Front X m] -> Path clear/blocked. 3. Constraint check: [Speed Limit] -> Adjusting parameters.",
            "mode": "normal" | "cautious" | "defense" | "emergency_stop",
            "action": "maintain" | "slow_down" | "avoid_left" | "avoid_right" | "land",
            "params": {"speed_limit": float, "collision_weight": float},
            "explanation": "Concise, tactical explanation for the operator dashboard."
        }

        LOGIC RULES:
        - IF 'primary_threat' risk_score > 0.6 THEN mode = 'defense', collision_weight > 2.0.
        - IF 'lidar' front_space < 2.0m THEN action = 'slow_down' OR 'avoid_...'.
        - IF constraints are violated, prioritize safety immediately.
        """

        # Kullanıcı Prompt'u
        user_prompt = f"""
        CURRENT SENSOR DATA: {context_json}
        MISSION GOAL: {mission}
        ACTIVE CONSTRAINTS: {constraints}
        
        EXECUTE CHAIN-OF-THOUGHT ANALYSIS AND DECIDE.
        """

        try:
            response = self.client.chat.completions.create(
                model=LOCAL_MODEL_NAME, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content
            
            # --- DEBUG: LLM NE DEDİ? (Terminalde gör) ---
            self.get_logger().info(f"🔍 HAM LLM ÇIKTISI: {content}")

            # --- MARKDOWN VE GÜRÜLTÜ TEMİZLİĞİ ---
            # 1. Regex ile ```json ... ``` arasını bulmaya çalış
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if match:
                clean_json = match.group(1)
            else:
                # 2. Regex bulamazsa klasik yöntem: İlk '{' ve son '}' arasını al
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    clean_json = content[start:end]
                else:
                    raise ValueError("JSON parantezleri bulunamadı.")

            return json.loads(clean_json)

        except Exception as e:
            error_msg = f"PARSE ERROR: {str(e)}"
            self.get_logger().error(error_msg)
            
            # Hata durumunda fail-safe dönüş
            return {
                "mode": "cautious", 
                "action": "hover",
                "explanation": f"LLM Parse Failed. Error: {str(e)}",
                "reasoning_trace": error_msg,
                "params": {}
            }

def main(args=None):
    rclpy.init(args=args)
    node = LLMStrategicNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()