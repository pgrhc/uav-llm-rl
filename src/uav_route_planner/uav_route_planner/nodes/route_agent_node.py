#!/usr/bin/env python3
"""
Route Agent Inference Node

Loads a trained PPO model and publishes waypoints at planning rate.
Replaces the heuristic_planner_node after training.

Input:
    /route/costmap_patch  (sensor_msgs/Image)
    /threat/state_vec     (std_msgs/Float32MultiArray)
    /odometry/filtered    (nav_msgs/Odometry)
    /goal_pose            (geometry_msgs/PoseStamped)
Output:
    /route/waypoint_desired (geometry_msgs/PoseStamped)
"""

import gymnasium as gym
import uav_route_planner.envs  # triggers register()
from stable_baselines3 import PPO
import rclpy
import os
import sys


def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)

    # Model path — pass as argument or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/RoutePPO/route_ppo_50000"

    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return

    print("--- ROTA AJANI BAŞLATILIYOR ---")

    # 1. Environment (provides ROS subscriptions + waypoint publishing)
    env = gym.make("RouteAgent-v0")
    obs, _ = env.reset()

    # 2. Load trained model
    try:
        model = PPO.load(model_path, env=env)
        print(f"Model yüklendi: {model_path}")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return

    # 3. Inference loop
    print("--- ROTA PLANLAMA AKTİF ---")
    try:
        while rclpy.ok():
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if info.get("success"):
                print("Hedefe ulaşıldı!")

            if terminated or truncated:
                obs, _ = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        print("Rota ajanı kapatılıyor...")
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
