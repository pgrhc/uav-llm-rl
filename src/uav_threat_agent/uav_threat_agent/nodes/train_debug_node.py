import gymnasium as gym
import uav_threat_agent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import rclpy

def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)

    print("=" * 60)
    print("🚀 THREAT AGENT EĞİTİMİ BAŞLIYOR (FIXED HYPERPARAMETERS)")
    print("=" * 60)

    # Klasörler
    models_dir = "models/PPO-v9-fixed"
    log_dir    = "logs/PPO-v9-fixed"

    for d in [models_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    # ─── 1. Ortam ──────────────────────────────────────────────────────────
    def make_env():
        env = gym.make('ThreatAgent-v9')
        env = Monitor(env, log_dir)
        return env

    env = DummyVecEnv([make_env])

    # VecNormalize: KRİTİK! Observation + Reward normalizasyonu
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=5.0
    )

    # ─── 2. Model (DÜZELTİLMİŞ HYPERPARAMETERS) ───────────────────────────
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,

        # ════════════════════════════════════════════════════════════════
        # DÜZELTILMIŞ HYPERPARAMETERS
        # ════════════════════════════════════════════════════════════════
        learning_rate=1e-4,     # ← 3e-4'ten DÜŞÜRÜLDÜ (KL divergence fix)
        n_steps=2048,
        batch_size=512,         # ← 256'dan ARTIRILDI (clip_fraction fix)
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,        # ← 0.2'den DÜŞÜRÜLDÜ (policy stability)
        ent_coef=0.08,          # ← 0.05'ten ARTIRILDI (exploration)
        vf_coef=0.5,
        max_grad_norm=0.5,
        # ════════════════════════════════════════════════════════════════

        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256, 128],
                vf=[256, 256, 128]
            )
        )
    )

    # ─── 3. Callbacks ──────────────────────────────────────────────────────
    # Her 20k adımda checkpoint kaydet (2048 çok sıktı)
    checkpoint_cb = CheckpointCallback(
        save_freq=2048,       # ← 2048'den ARTIRILDI
        save_path=models_dir,
        name_prefix="ppo_threat",
        save_vecnormalize=True
    )

    # ─── 4. Eğitim ─────────────────────────────────────────────────────────
    TOTAL_TIMESTEPS = 204_800  # ← 204k'dan ARTIRILDI (minimum 1M)

    print(f"\n📊 Eğitim Ayarları:")
    print(f"  • Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  • Learning Rate: {model.learning_rate}")
    print(f"  • Batch Size: {model.batch_size}")
    print(f"  • Clip Range: {model.clip_range}")
    print(f"  • Entropy Coef: {model.ent_coef}")
    print(f"  • Checkpoint Frequency: Every 20,000 steps")
    print(f"\n🎯 Hedef Metrikler (1M timestep sonunda):")
    print(f"  • ep_rew_mean: -300 ile 0 arası")
    print(f"  • approx_kl: 0.02-0.03")
    print(f"  • clip_fraction: 0.15-0.25")
    print(f"  • std: 1.5-2.5")
    print("=" * 60)
    print()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_cb,
        tb_log_name="PPO_v10_fixed",
        progress_bar=True
    )

    # ─── 5. Son Kayıt ──────────────────────────────────────────────────────
    model.save(f"{models_dir}/final_model")
    env.save(f"{models_dir}/vec_normalize.pkl")

    print("\n" + "=" * 60)
    print("✅ Eğitim tamamlandı!")
    print(f"   Model: {models_dir}/final_model.zip")
    print(f"   VecNormalize: {models_dir}/vec_normalize.pkl")
    print("=" * 60)

    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()