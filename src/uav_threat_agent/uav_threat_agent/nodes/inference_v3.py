import os
import json
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import uav_threat_agent


class ThreatPublisher(Node):
    def __init__(self):
        super().__init__('threat_agent_publisher')

        self.score_pub = self.create_publisher(
            Float32MultiArray, '/threat/output_scores', 10
        )
        self.info_pub = self.create_publisher(
            String, '/threat/detailed_info', 10
        )


def make_env():
    env = gym.make('ThreatAgent-v13')
    return env


def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)

    print("--- SAC ASİMETRİK AJAN YÜKLENİYOR... ---")


    model_path = "/home/ubuntu/Desktop/ros2_env/models/SAC-AsymAC-20260316-170504/final_model.zip"
    vecnorm_path = "/home/ubuntu/Desktop/ros2_env/models/SAC-AsymAC-20260316-170504/vec_normalize.pkl"

    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return

    if not os.path.exists(vecnorm_path):
        print(f"HATA: VecNormalize dosyası bulunamadı: {vecnorm_path}")
        return


    env = DummyVecEnv([make_env])

    try:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print("--- VecNormalize başarıyla yüklendi ---")
    except Exception as e:
        print(f"VecNormalize yükleme hatası: {e}")
        return

    obs = env.reset()

    try:
        model = SAC.load(model_path, env=env)
        print("--- MODEL BAŞARIYLA YÜKLENDİ ---")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return

    threat_pub = ThreatPublisher()

    try:
        while rclpy.ok():
            action, _state = model.predict(obs, deterministic=True)

            score_msg = Float32MultiArray()
            score_msg.data = action[0].tolist() if isinstance(action, np.ndarray) and action.ndim > 1 else action.tolist()
            threat_pub.score_pub.publish(score_msg)
            obs, reward, done, info = env.step(action)
            info0 = info[0] if isinstance(info, list) and len(info) > 0 else {}

            if "top_threats" in info0:
                json_msg = String()
                json_msg.data = json.dumps(info0["top_threats"])
                threat_pub.info_pub.publish(json_msg)
            if done[0] if isinstance(done, np.ndarray) else done:
                obs = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        print("Kapatılıyor...")
        env.close()
        threat_pub.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()