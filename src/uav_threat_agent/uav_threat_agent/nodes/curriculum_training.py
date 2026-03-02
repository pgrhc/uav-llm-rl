#!/usr/bin/env python3
"""
CURRICULUM LEARNING - DIRECT IMPORT VERSION
gym.make() kullanmadan direkt ThreatAgentEnv import eder
"""
import os
import time
import json
import signal
import sys
from datetime import datetime

# Direct import!
from uav_threat_agent.envs.curriculum_env import ThreatAgentEnv

import rclpy
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


# ══════════════════════════════════════════════════════════════════════════════
# GLOBALS
# ══════════════════════════════════════════════════════════════════════════════
_model = None
_env = None
_models_dir = None


def signal_handler(sig, frame):
    print("\n\n🛑 Ctrl+C algılandı! Model kaydediliyor...")
    try:
        if _model is not None and _models_dir is not None:
            save_path = os.path.join(_models_dir, "interrupted_model")
            _model.save(save_path)
            print(f"✅ Model kaydedildi: {save_path}.zip")
    except Exception as e:
        print(f"⚠️  Kayıt hatası: {e}")
    finally:
        if _env is not None:
            _env.close()
        if rclpy.ok():
            rclpy.shutdown()
        print("\n👋 Güvenli şekilde kapatıldı.")
        sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

class StageProgressReporter(BaseCallback):
    def __init__(self, stage: int, stage_timesteps: int, report_freq: int = 5_000):
        super().__init__()
        self.stage = stage
        self.stage_timesteps = stage_timesteps
        self.report_freq = report_freq
        self.start_time = time.time()
        self.stage_start_timesteps = 0

    def _on_training_start(self):
        self.stage_start_timesteps = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps % self.report_freq == 0:
            now = time.time()
            elapsed = now - self.start_time
            stage_progress = self.num_timesteps - self.stage_start_timesteps
            progress_pct = (stage_progress / self.stage_timesteps) * 100
            
            print(f"\n⏱️  STAGE {self.stage}: {stage_progress:,} / {self.stage_timesteps:,} ({progress_pct:.1f}%)")
            print(f"   Total: {self.num_timesteps:,} timesteps | Time: {elapsed/60:.1f}m")
        return True


class StageMetricsLogger(BaseCallback):
    def __init__(self, stage: int, log_freq: int = 10_000):
        super().__init__()
        self.stage = stage
        self.log_freq = log_freq
        self.stage_rewards = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                self.stage_rewards.append(ep_info.get("r", 0.0))

        if self.num_timesteps % self.log_freq == 0 and len(self.stage_rewards) > 0:
            mean_rew = float(np.mean(self.stage_rewards[-100:]))
            self.logger.record(f"stage{self.stage}/ep_rew_mean", mean_rew)
            print(f"\n📊 STAGE {self.stage} Reward Mean: {mean_rew:.2f}")
        return True


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_stage(stage, model, env, timesteps, models_dir, log_dir):
    global _model, _env
    _model = model
    _env = env
    
    print(f"\n{'=' * 70}")
    print(f"🎓 STAGE {stage}: {timesteps:,} timesteps")
    print(f"{'=' * 70}")
    
    if stage == 1:
        print("   Goal: Learn basic scoring (Person=high, Unknown=low)")
    elif stage == 2:
        print("   Goal: Learn distance-based scoring")
    else:
        print("   Goal: Full complexity optimization")
    print()
    
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=models_dir,
        name_prefix=f"stage{stage}",
        verbose=1,
    )
    
    progress_cb = StageProgressReporter(stage, timesteps, 5_000)
    metrics_cb = StageMetricsLogger(stage, 10_000)
    
    callbacks = CallbackList([checkpoint_cb, progress_cb, metrics_cb])
    
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=(stage == 1),
        progress_bar=True,
    )
    
    stage_path = os.path.join(models_dir, f"stage{stage}_complete")
    model.save(stage_path)
    print(f"\n✅ Stage {stage} complete: {stage_path}.zip\n")
    
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    global _model, _env, _models_dir
    
    if not rclpy.ok():
        rclpy.init(args=args)

    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 70)
    print("🎓 CURRICULUM LEARNING (DIRECT IMPORT)")
    print("=" * 70)

    SEED = 42
    set_random_seed(SEED)
    np.random.seed(SEED)

    run_name = f"PPO-CURR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    models_dir = f"models/{run_name}"
    log_dir = f"logs/{run_name}"
    _models_dir = models_dir
    
    for d in [models_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    meta = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "total_timesteps": 200_000,
        "stages": {
            "1": {"timesteps": 50_000, "goal": "Basic scoring"},
            "2": {"timesteps": 50_000, "goal": "Distance-based"},
            "3": {"timesteps": 100_000, "goal": "Full complexity"}
        }
    }
    with open(os.path.join(log_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1
    # ═══════════════════════════════════════════════════════════════════════
    print("\n🎓 Creating STAGE 1 environment...")
    
    def make_env_stage1():
        # DIRECT INSTANTIATION!
        env = ThreatAgentEnv(curriculum_stage=1)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor_s1.csv"))
        try:
            env.reset(seed=SEED)
        except:
            pass
        return env

    env = DummyVecEnv([make_env_stage1])
    _env = env

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=SEED,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )
    _model = model

    model = train_stage(1, model, env, 50_000, models_dir, log_dir)
    env.close()

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2
    # ═══════════════════════════════════════════════════════════════════════
    print("\n🎓 Creating STAGE 2 environment...")
    
    def make_env_stage2():
        env = ThreatAgentEnv(curriculum_stage=2)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor_s2.csv"))
        return env

    env = DummyVecEnv([make_env_stage2])
    _env = env
    model.set_env(env)

    model = train_stage(2, model, env, 50_000, models_dir, log_dir)
    env.close()

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 3
    # ═══════════════════════════════════════════════════════════════════════
    print("\n🎓 Creating STAGE 3 environment...")
    
    def make_env_stage3():
        env = ThreatAgentEnv(curriculum_stage=3)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor_s3.csv"))
        return env

    env = DummyVecEnv([make_env_stage3])
    _env = env
    model.set_env(env)

    model = train_stage(3, model, env, 100_000, models_dir, log_dir)

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL
    # ═══════════════════════════════════════════════════════════════════════
    final_path = os.path.join(models_dir, "final_200k")
    model.save(final_path)

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Total: 200k timesteps")
    print(f"Model: {final_path}.zip")
    print(f"TensorBoard: tensorboard --logdir={log_dir}")
    print("=" * 70)

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()