#!/usr/bin/env python3
"""
Route Agent Inference Node  (Faz 1 v2)

Loads a trained PPO model + VecNormalize stats and publishes waypoints.

Input:
    /route/costmap_patch    (sensor_msgs/Image)
    /threat/state_vec       (std_msgs/Float32MultiArray)   74-dim
    /threat/target_scores   (std_msgs/Float32MultiArray)   5-dim
    /odometry/filtered      (nav_msgs/Odometry)
    /goal_pose              (geometry_msgs/PoseStamped)
    /plan                   (nav_msgs/Path)                A* reference
Output:
    /route/waypoint_desired (geometry_msgs/PoseStamped)
"""

import gymnasium as gym
import uav_route_planner.envs  # triggers register()
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import rclpy
import os
import sys


def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/RoutePPO_v2/route_ppo_v2_500000"

    vecnorm_path = model_path.replace("route_ppo_v2_", "vecnorm_v2_") + ".pkl"

    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return

    print("--- ROTA AJANI BAŞLATILIYOR ---")

    vec_env = DummyVecEnv([lambda: gym.make("RouteAgent-v0")])

    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, vec_env)
        env.training = False
        env.norm_reward = False
        print(f"VecNormalize yüklendi: {vecnorm_path}")
    else:
        env = vec_env
        print("VecNormalize bulunamadı, normalizasyonsuz çalışılıyor.")

    try:
        model = PPO.load(model_path, env=env)
        print(f"Model yüklendi: {model_path}")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return

    print("--- ROTA PLANLAMA AKTİF ---")
    obs = env.reset()
    try:
        while rclpy.ok():
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            if isinstance(info, list) and info[0].get("success"):
                print("Hedefe ulaşıldı!")

            if isinstance(done, list):
                done = done[0]
            if done:
                obs = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        print("Rota ajanı kapatılıyor...")
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
