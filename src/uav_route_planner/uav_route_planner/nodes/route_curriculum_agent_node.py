#!/usr/bin/env python3
"""
Route Curriculum Agent Inference Node (Symmetric PPO)

Loads trained PPO model + VecNormalize.
Uses RouteCurriculumAgent-v0 env.

Input/Output: Same as route_agent_node
"""

import numpy as np
import gymnasium as gym
import uav_route_planner.envs  # noqa: F401 — triggers register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import rclpy
import os
import sys


def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/route_curriculum/final_model"

    base = os.path.dirname(model_path)
    name = os.path.basename(model_path.rstrip(".zip"))
    if name == "final_model":
        vecnorm_path = os.path.join(base, "vec_normalize_final.pkl")
    else:
        # Checkpoint: route_curriculum_50000 -> try vecnorm_50000.pkl or route_curriculum_50000_vecnormalize.pkl
        step = name.split("_")[-1] if "_" in name else ""
        vecnorm_path = os.path.join(base, f"vecnorm_{step}.pkl")
        if not os.path.exists(vecnorm_path):
            vecnorm_path = os.path.join(base, f"{name}_vecnormalize.pkl")
        if not os.path.exists(vecnorm_path):
            vecnorm_path = os.path.join(base, "vec_normalize_final.pkl")

    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"HATA: Model dosyasi bulunamadi: {model_path}")
        return

    print("--- ROTA CURRICULUM AJANI (PPO) BASLATILIYOR ---")

    vec_env = DummyVecEnv([lambda: gym.make("RouteCurriculumAgent-v0")])

    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, vec_env)
        env.training = False
        env.norm_reward = False
        print(f"VecNormalize yuklendi: {vecnorm_path}")
    else:
        print(
            "UYARI: VecNormalize bulunamadi! Model norm_obs=True ile egitildi. "
            "Normalizasyonsuz inference beklenmedik davranis uretir. "
            f"VecNormalize dosyasi beklenen konumda olmali: {vecnorm_path}"
        )
        sys.exit(1)

    try:
        model = PPO.load(
            model_path,
            env=env,
        )
        print(f"Model yuklendi: {model_path}")
    except Exception as e:
        print(f"Model yukleme hatasi: {e}")
        return

    print("--- ROTA PLANLAMA AKTIF ---")
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    try:
        while rclpy.ok():
            action, _state = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            if isinstance(infos, list) and infos and infos[0].get("success"):
                print("Hedefe ulasildi!")

            if isinstance(dones, (list, np.ndarray)):
                done = bool(dones[0])
            else:
                done = bool(dones)
            if done:
                result = env.reset()
                obs = result[0] if isinstance(result, tuple) else result

    except KeyboardInterrupt:
        pass
    finally:
        print("Rota curriculum ajanı kapatılıyor...")
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
