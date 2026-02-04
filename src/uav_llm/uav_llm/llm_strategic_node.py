import rclpy
from rclpy.node import Node
import json
import time

# Az önce tanımladığımız ve derlediğimiz servis yapısı
from fusion_msgs.srv import StrategicAdvice

class LLMStrategicNode(Node):
    def __init__(self):
        super().__init__('llm_strategic_node')
        
        # Servis Sunucusu (Service Server) oluşturuluyor
        # Bu servis 'llm/plan_and_explain' adıyla yayına başlar.
        self.srv = self.create_service(
            StrategicAdvice, 
            'llm/plan_and_explain', 
            self.handle_strategic_advice
        )
        
        self.get_logger().info('🧠 LLM Stratejik Katmanı (Servis) Hazır ve Bekliyor...')

    def handle_strategic_advice(self, request, response):
        """
        Mission Manager'dan gelen isteği karşılayan fonksiyon.
        """
        start_time = time.time()
        self.get_logger().info(f'📨 İstek Alındı: {request.mission_goal}')

        # --- BURASI İLERİDE GERÇEK LLM'E BAĞLANACAK ---
        # Şimdilik "Dummy" (Sahte) cevap döndürelim.
        
        try:
            # 1. Gelen veriyi (Simülasyon) analiz et
            # summary = json.loads(request.system_summary_json)
            
            # 2. Sahte bir 'Düşünme' süresi (LLM gecikmesini simüle etmek için)
            time.sleep(1.0) 

            # 3. Örnek Cevap Oluştur
            dummy_advice = {
                "mode": "cautious",
                "reward_weights": {"collision": 1.5, "progress": 0.8},
                "planning_hint": "avoid_unknown_areas"
            }
            
            dummy_explanation = "Bilinmeyen bir alana girildiği için tedbir modu aktif edildi."

            # 4. Response'u doldur
            response.advice_json = json.dumps(dummy_advice)
            response.explanation_text = dummy_explanation
            response.success = True
            
        except Exception as e:
            self.get_logger().error(f'Hata: {str(e)}')
            response.success = False
            response.explanation_text = "Internal LLM Node Error"
        
        process_time = time.time() - start_time
        self.get_logger().info(f'✅ Cevap Üretildi ({process_time:.2f}s): {response.explanation_text}')
        
        return response

def main(args=None):
    rclpy.init(args=args)
    node = LLMStrategicNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()