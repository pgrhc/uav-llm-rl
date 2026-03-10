#!/usr/bin/env python3
"""
Training script for the Route Planning Agent  (Faz 1 v2 — PPO + Gazebo).

Düzeltmeler (v2):
    - learning_rate:  3e-4 → 1e-4   (KL divergence çok yüksekti)
    - n_steps:        1024 → 2048    (daha stabil gradient)
    - batch_size:     64   → 256     (daha az varyans)
    - ent_coef:       0.01 → 0.05   (keşif çökmesini engelle)
    - clip_range:     0.2  → 0.3    (clip fraction çok yüksekti)
    - VecNormalize eklendi           (observation/reward normalizasyonu)
    - total_timesteps: 200K → 500K

Prerequisites:
    1. Gazebo simulation running with the drone + maze
    2. Nav2 + SLAM providing /local_costmap/costmap
    3. CostmapPatchNode publishing /route/costmap_patch
    4. ThreatEncoderV2 publishing /threat/state_vec  (74-dim)
    5. ThreatTargetNode publishing /threat/target_scores
    6. Odometry on /odometry/filtered
    7. A* path on /plan  (auto_maze_navigator)
    8. follow_path running  (route agent waypoint'lerini PX4'e iletir)

Usage:
    ros2 run uav_route_planner train_route_node
"""

import gymnasium as gym
import uav_route_planner.envs  # triggers register()
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import rclpy
import os


def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)

    print("--- ROTA AJANI EĞİTİMİ BAŞLATILIYOR (Faz 1 v2 — 500K) ---")

    vec_env = DummyVecEnv([lambda: gym.make("RouteAgent-v0")])
    env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    models_dir = "models/RoutePPO_v2"
    log_dir = "logs/route_v2"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    from uav_route_planner.networks.route_extractor import RouteCombinedExtractor

    policy_kwargs = dict(
        features_extractor_class=RouteCombinedExtractor,
        net_arch=[256, 128],
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.3,
        ent_coef=0.05,
        policy_kwargs=policy_kwargs,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=10000,
        save_path=models_dir,
        name_prefix="route_ppo_v2",
    )

    TOTAL_TIMESTEPS = 500_000
    SAVE_INTERVAL = 50_000

    steps_done = 0
    while steps_done < TOTAL_TIMESTEPS:
        chunk = min(SAVE_INTERVAL, TOTAL_TIMESTEPS - steps_done)
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            tb_log_name="RoutePPO_v2",
            callback=checkpoint_cb,
        )
        steps_done += chunk

        model.save(f"{models_dir}/route_ppo_v2_{steps_done}")
        env.save(f"{models_dir}/vecnorm_v2_{steps_done}.pkl")
        print(f"Model kaydedildi: {steps_done}/{TOTAL_TIMESTEPS}")

    print("--- EĞİTİM TAMAMLANDI ---")
    env.close()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
