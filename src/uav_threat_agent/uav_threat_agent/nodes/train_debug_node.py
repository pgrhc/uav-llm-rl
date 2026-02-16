import gymnasium as gym
import uav_threat_agent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
import os
import rclpy

def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)

    print("--- EĞİTİM BAŞLATILIYOR ---")

    # Klasörler
    models_dir = "models/PPO-v7-scratch"
    log_dir    = "logs/PPO-v7-scratch"
    best_dir   = "models/PPO-v7-best"

    for d in [models_dir, log_dir, best_dir]:
        os.makedirs(d, exist_ok=True)

    # ─── 1. Ortam ──────────────────────────────────────────────
    # Monitor: Her episode reward/length loglanır
    def make_env():
        env = gym.make('ThreatAgent-v7')
        env = Monitor(env, log_dir)
        return env

    env      = DummyVecEnv([make_env])
    # eval_env = DummyVecEnv([make_env])

    # VecNormalize: Observation + Reward normalizasyonu
    # norm_reward=True eğitimi çok stabilleştirir
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,   # Aşırı değerleri kırp
        clip_reward=5.0
    )
    # eval_env = VecNormalize(
    #     eval_env,
    #     norm_obs=True,
    #     norm_reward=False,  # Eval'da reward normalize etme (gerçek değeri gör)
    #     clip_obs=10.0,
    #     training=False      # Eval env istatistik güncellemesin
    # )

    # ─── 2. Model ──────────────────────────────────────────────
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,

        # Hyperparameters
        learning_rate=3e-4,
        n_steps=2048,        # Her update için toplanan adım (artırıldı)
        batch_size=256,      # 2048 / 256 = 8 mini-batch (verimli)
        n_epochs=10,
        gamma=0.99,          # Uzun vadeli ödül ağırlığı
        gae_lambda=0.95,
        clip_range=0.2,
        
        # Entropy: Keşfetmeyi teşvik et, zamanla azalt
        ent_coef=0.05,       # Başlangıçta yüksek (keşif)
        
        # Ağ mimarisi
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256, 128],  # Policy network (actor)
                vf=[256, 256, 128]   # Value network (critic) — ayrı tut!
            )
        )
    )

    # ─── 3. Callbacks ──────────────────────────────────────────
    
    # Her 10.000 adımda kaydet
    checkpoint_cb = CheckpointCallback(
        save_freq=2048,
        save_path=models_dir,
        name_prefix="ppo_threat",
        save_vecnormalize=True  # ← Normalizasyon istatistiklerini de kaydet!
    )

    # Her 20.000 adımda eval yap, en iyi modeli sakla
    # eval_cb = EvalCallback(
    #     eval_env,
    #     best_model_save_path=best_dir,
    #     log_path=log_dir,
    #     eval_freq=2048,
    #     n_eval_episodes=5,
    #     deterministic=True,
    #     render=False
    # )

    # callbacks = CallbackList([checkpoint_cb, eval_cb])

    # ─── 4. Eğitim ─────────────────────────────────────────────
    TOTAL_TIMESTEPS = 24_576  # Minimum! Gerekirse artır.

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_cb,
        tb_log_name="PPO_scratch",
        progress_bar=True
    )

    # ─── 5. Son Kayıt ───────────────────────────────────────────
    model.save(f"{models_dir}/final_model")
    env.save(f"{models_dir}/vec_normalize.pkl")  # ← ŞART! Deploy'da da lazım.
    print("Eğitim tamamlandı. Final model kaydedildi.")

    env.close()
    # eval_env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()