#!/usr/bin/env python3
"""
Training script for the Route Planning Agent (PPO + Gazebo).

Prerequisites:
    1. Gazebo simulation running with the drone
    2. Nav2 + SLAM providing /local_costmap/costmap
    3. CostmapPatchNode publishing /route/costmap_patch
    4. ThreatEncoderNode publishing /threat/state_vec
    5. Odometry on /odometry/filtered
    6. A goal published on /goal_pose

Usage:
    python3 train_route_node.py
"""

import gymnasium as gym
import uav_route_planner.envs  # triggers register()
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import rclpy
import os


def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)

    print("--- ROTA AJANI EĞİTİMİ BAŞLATILIYOR ---")

    # 1. Create environment
    env = gym.make("RouteAgent-v0")

    # 2. Directories
    models_dir = "models/RoutePPO"
    log_dir = "logs/route"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 3. Custom feature extractor
    from uav_route_planner.networks.route_extractor import RouteCombinedExtractor

    policy_kwargs = dict(
        features_extractor_class=RouteCombinedExtractor,
        net_arch=[256, 128],
    )

    # 4. PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
    )

    # 5. Checkpoint callback
    checkpoint_cb = CheckpointCallback(
        save_freq=2048,
        save_path=models_dir,
        name_prefix="route_ppo",
    )

    # 6. Training loop
    TOTAL_TIMESTEPS = 5_000
    SAVE_INTERVAL = 2_500

    steps_done = 0
    while steps_done < TOTAL_TIMESTEPS:
        chunk = min(SAVE_INTERVAL, TOTAL_TIMESTEPS - steps_done)
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            tb_log_name="RoutePPO",
            callback=checkpoint_cb,
        )
        steps_done += chunk
        save_path = f"{models_dir}/route_ppo_{steps_done}"
        model.save(save_path)
        print(f"Model kaydedildi: {save_path} ({steps_done}/{TOTAL_TIMESTEPS})")

    print("--- EĞİTİM TAMAMLANDI ---")
    env.close()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
