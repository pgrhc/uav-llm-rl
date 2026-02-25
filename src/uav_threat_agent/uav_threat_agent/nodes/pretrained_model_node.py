#!/usr/bin/env python3
import os
import time
import json
import signal
import sys
from datetime import datetime

import gymnasium as gym
import uav_threat_agent

import rclpy
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO



# ══════════════════════════════════════════════════════════════════════════════
# GLOBALS FOR SIGNAL HANDLER
# ══════════════════════════════════════════════════════════════════════════════
_model = None
_env = None
_models_dir = None
_training_logger = None


def signal_handler(sig, frame):
    """Ctrl+C ile güvenli shutdown."""
    print("\n\n🛑 Ctrl+C algılandı! Model kaydediliyor...")
    try:
        if _model is not None and _models_dir is not None:
            save_path = os.path.join(_models_dir, "interrupted_model")
            _model.save(save_path)
            print(f"✅ Model kaydedildi: {save_path}.zip")
        
        if _env is not None and _models_dir is not None:
            vec_path = os.path.join(_models_dir, "vec_normalize_interrupted.pkl")
            _env.save(vec_path)
            print(f"✅ VecNormalize kaydedildi: {vec_path}")
        
        if _training_logger is not None:
            _training_logger._flush()
            print(f"✅ Training log kaydedildi")
            
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
# CUSTOM CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

class LazyStrategyDetector(BaseCallback):
    def __init__(self, check_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.actions_buffer = []

    def _on_step(self) -> bool:
        actions = self.locals.get("actions", None)
        if actions is None:
            return True

        if len(actions.shape) == 2:
            action = actions[0]
        else:
            action = actions

        self.actions_buffer.append(action.copy())

        if self.num_timesteps % self.check_freq == 0:
            self._analyze_and_report()
            self.actions_buffer = []

        return True

    def _analyze_and_report(self):
        if len(self.actions_buffer) == 0:
            return

        actions = np.array(self.actions_buffer)

        mean_action = float(np.mean(actions))
        action_std = float(np.std(actions))
        zero_ratio = float(np.sum(actions < 0.1) / actions.size)
        high_ratio = float(np.sum(actions > 0.5) / actions.size)

        self.logger.record("custom/mean_action", mean_action)
        self.logger.record("custom/action_std", action_std)
        self.logger.record("custom/zero_ratio", zero_ratio)
        self.logger.record("custom/high_ratio", high_ratio)

        warnings = []
        if mean_action < 0.15:
            warnings.append("⚠️  LAZY: Mean action < 0.15")
        if action_std < 0.1:
            warnings.append("⚠️  NO DIVERSITY: Std < 0.1")
        if zero_ratio > 0.7:
            warnings.append("⚠️  TOO MANY ZEROS: >70%")
        if high_ratio < 0.05:
            warnings.append("⚠️  NO HIGH SCORES: <5%")

        print("\n" + "=" * 70)
        print(f"📊 ACTION CHECK @ {self.num_timesteps:,} timesteps")
        print("=" * 70)
        print(f"  Mean: {mean_action:.3f} | Std: {action_std:.3f}")
        print(f"  Zero: {zero_ratio:.1%} | High: {high_ratio:.1%}")

        if warnings:
            print("\n🚨 ALARMLAR:")
            for w in warnings:
                print(f"  {w}")
        else:
            print("\n✅ Sağlıklı görünüyor")
        print("=" * 70 + "\n")


class ProgressReporter(BaseCallback):
    def __init__(self, target_steps: int, report_freq: int = 5_000):
        super().__init__()
        self.target_steps = int(target_steps)
        self.report_freq = int(report_freq)
        self.start_time = time.time()
        self.last_report_time = self.start_time

    def _on_step(self) -> bool:
        if self.num_timesteps % self.report_freq == 0:
            self._print_progress()
        return True

    def _print_progress(self):
        now = time.time()
        elapsed_total = now - self.start_time
        elapsed_interval = now - self.last_report_time

        fps = self.report_freq / elapsed_interval if elapsed_interval > 0 else 0.0

        if self.num_timesteps > 0:
            sec_per_step = elapsed_total / self.num_timesteps
            remaining_steps = max(0, self.target_steps - self.num_timesteps)
            remaining_sec = remaining_steps * sec_per_step
            hours = int(remaining_sec // 3600)
            mins = int((remaining_sec % 3600) // 60)
        else:
            hours, mins = 0, 0

        print(f"\n⏱️  {self.num_timesteps:,} / {self.target_steps:,} timesteps")
        print(f"   FPS: {fps:.1f} | Kalan: ~{hours}h {mins}m")

        self.last_report_time = now


class EnvironmentMetricsLogger(BaseCallback):
    def __init__(self, log_freq: int = 10_000):
        super().__init__()
        self.log_freq = int(log_freq)
        self.person_count = 0
        self.unknown_count = 0
        self.high_score_count = 0
        self.total_objects = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                top = info.get("top_threats", None)
                if top is None:
                    continue

                for threat in top:
                    self.total_objects += 1
                    cls = threat.get("cls", "")
                    score = float(threat.get("score", 0.0))

                    if cls == "Person":
                        self.person_count += 1
                    elif cls == "Unknown":
                        self.unknown_count += 1

                    if score > 0.5:
                        self.high_score_count += 1

        if (self.num_timesteps % self.log_freq == 0) and (self.total_objects > 0):
            person_ratio = self.person_count / self.total_objects
            high_score_ratio = self.high_score_count / self.total_objects

            self.logger.record("env/person_detection_ratio", person_ratio)
            self.logger.record("env/high_score_ratio", high_score_ratio)

            print(f"\n📈 ENV METRICS @ {self.num_timesteps:,}")
            print(f"   Person Detection: {person_ratio:.1%}")
            print(f"   High Score Ratio: {high_score_ratio:.1%}")

            self.person_count = 0
            self.unknown_count = 0
            self.high_score_count = 0
            self.total_objects = 0

        return True


class TrainingLogger(BaseCallback):
    def __init__(self, log_file: str, log_freq: int = 5_000, flush_every: int = 10_000):
        super().__init__()
        self.log_file = log_file
        self.log_freq = int(log_freq)
        self.flush_every = int(flush_every)
        self.logs = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            ep_buf = list(self.model.ep_info_buffer) if hasattr(self.model, "ep_info_buffer") else []

            if len(ep_buf) > 0:
                ep_rewards = [e.get("r", 0.0) for e in ep_buf]
                ep_lens = [e.get("l", 0.0) for e in ep_buf]
                ep_rew_mean = float(np.mean(ep_rewards))
                ep_len_mean = float(np.mean(ep_lens))
            else:
                ep_rew_mean = 0.0
                ep_len_mean = 0.0

            # PPO metrikleri (mevcut ise)
            try:
                approx_kl = float(self.model.logger.name_to_value.get("train/approx_kl", 0.0))
                clip_frac = float(self.model.logger.name_to_value.get("train/clip_fraction", 0.0))
            except (AttributeError, KeyError):
                approx_kl = 0.0
                clip_frac = 0.0

            log_entry = {
                "timesteps": int(self.num_timesteps),
                "time": datetime.now().isoformat(),
                "ep_rew_mean": ep_rew_mean,
                "ep_len_mean": ep_len_mean,
                "approx_kl": approx_kl,
                "clip_fraction": clip_frac,
            }
            self.logs.append(log_entry)

            if self.num_timesteps % self.flush_every == 0:
                self._flush()

        return True

    def _flush(self):
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.logs, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"⚠️  JSON log yazılamadı: {e}")
        except Exception as e:
            print(f"⚠️  Beklenmeyen hata (TrainingLogger): {e}")


# ══════════════════════════════════════════════════════════════════════════════
# EARLY EVALUATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def early_lazy_check(model, env, n_episodes: int = 5, max_steps: int = 2048):
    """
    İlk 20k sonunda lazy strategy kontrolü.
    
    FIX: mean_action normalize edilmiş olabileceği için SADECE zero_ratio'ya bak!
    """
    print("\n" + "=" * 70)
    print("🔍 EARLY LAZY CHECK @ 20k timesteps")
    print("=" * 70)

    all_actions = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = np.array([False])
        steps = 0

        while (not done[0]) and (steps < max_steps):
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(action[0].copy())
            obs, _, done, _ = env.step(action)
            steps += 1

    actions = np.array(all_actions)
    
    # ═══════════════════════════════════════════════════════════════════════
    # FIX: mean_action normalize edilmiş olabilir, güvenilmez!
    # ═══════════════════════════════════════════════════════════════════════
    # mean_action = float(np.mean(actions))  # ← KULLANMA!
    
    zero_ratio = float(np.sum(actions < 0.1) / actions.size)
    high_ratio = float(np.sum(actions > 0.5) / actions.size)
    
    # print(f"  Mean Action: {mean_action:.3f}")  # ← Misleading, gösterme
    print(f"  Zero Ratio:  {zero_ratio:.1%}")
    print(f"  High Ratio:  {high_ratio:.1%}")

    # ═══════════════════════════════════════════════════════════════════════
    # SADECE zero_ratio ve high_ratio'ya bak!
    # ═══════════════════════════════════════════════════════════════════════
    is_lazy = (zero_ratio > 0.75) or (high_ratio < 0.03)
    
    if is_lazy:
        print("\n🚨 LAZY STRATEGY DETECTED!")
        print("  → Zero ratio >75% VEYA high ratio <3%")
        print("  → Eğitim DURDURULMALI!")
        return False
    else:
        print("\n✅ Action distribution OK")
        print(f"  → Zero: {zero_ratio:.1%} (<75% ✓)")
        print(f"  → High: {high_ratio:.1%} (>3% ✓)")
        print("  → Devam edilebilir.")
        return True

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING
# ══════════════════════════════════════════════════════════════════════════════

#!/usr/bin/env python3


# Önceki training script'teki callback'leri import et
# (LazyStrategyDetector, ProgressReporter, etc.)
# ... (callback kodlarını buraya kopyala)


def main():
    global _model, _env, _models_dir, _training_logger
    
    if not rclpy.ok():
        rclpy.init()

    print("=" * 70)
    print("🔄 CHECKPOINT'TEN DEVAM ETTİRME")
    print("=" * 70)

    signal.signal(signal.SIGINT, signal_handler)

    # ═══════════════════════════════════════════════════════════════════════
    # 1. CHECKPOINT AYARLARI (BURAYA YOLU YAZ)
    # ═══════════════════════════════════════════════════════════════════════
    # ✏️ Burayı senin checkpoint'ine göre değiştir:
    OLD_RUN_NAME = "PPO-11-20260225-071128"  # ← Klasör adı
    CHECKPOINT_FILE = "ppo_threat_40960_steps"  # ← .zip olmadan
    
    models_base = "models"
    old_models_dir = os.path.join(models_base, OLD_RUN_NAME)
    
    # Checkpoint path'leri
    checkpoint_path = os.path.join(old_models_dir, f"{CHECKPOINT_FILE}.zip")
    vecnorm_path = os.path.join(old_models_dir, f"{CHECKPOINT_FILE}_vecnormalize.pkl")
    
    # Kontrol et
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Checkpoint bulunamadı!")
        print(f"   Aranan: {checkpoint_path}")
        print(f"\n💡 Mevcut checkpoint'leri görmek için:")
        print(f"   ls {old_models_dir}/*.zip")
        return
    
    print(f"\n📂 Orjinal run: {OLD_RUN_NAME}")
    print(f"   Checkpoint: {CHECKPOINT_FILE}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # 2. YENİ KLASÖR OLUŞTUR
    # ═══════════════════════════════════════════════════════════════════════
    resume_suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_run_name = f"{OLD_RUN_NAME}_resume_{resume_suffix}"
    
    models_dir = os.path.join(models_base, new_run_name)
    log_dir = f"logs/{new_run_name}"
    _models_dir = models_dir
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n📂 Resume run: {new_run_name}")
    print(f"   Models: {models_dir}")
    print(f"   Logs:   {log_dir}")
    print(f"\n✅ Override riski YOK")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. ENVIRONMENT
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n🔧 Environment oluşturuluyor...")
    
    def make_env():
        env = gym.make("ThreatAgent-v11")
        env = Monitor(env, filename=os.path.join(log_dir, "monitor_resume.csv"))
        return env

    env = DummyVecEnv([make_env])
    
    # VecNormalize yükle
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        print(f"   ✅ VecNormalize yüklendi: {vecnorm_path}")
    else:
        print(f"   ⚠️  VecNormalize bulunamadı, yenisi oluşturuluyor")
        env = VecNormalize(env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=5.0)
    
    _env = env

    # ═══════════════════════════════════════════════════════════════════════
    # 4. MODEL YÜKLEME
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n📥 Model yükleniyor...")
    print(f"   Path: {checkpoint_path}")
    
    model = PPO.load(checkpoint_path, env=env)
    _model = model
    
    print(f"   ✅ Model yüklendi")
    print(f"\n📊 Mevcut durum:")
    print(f"   Timesteps: {model.num_timesteps:,}")
    print(f"   → Hedef: 204,800")
    print(f"   → Kalan: {204_800 - model.num_timesteps:,}")

    # ═══════════════════════════════════════════════════════════════════════
    # 5. CALLBACKS
    # ═══════════════════════════════════════════════════════════════════════
    TOTAL_TIMESTEPS = 204_800
    remaining_timesteps = max(0, TOTAL_TIMESTEPS)
    
    checkpoint_cb = CheckpointCallback(
        save_freq=4096,
        save_path=models_dir,
        name_prefix="ppo_threat",
        save_vecnormalize=True,
        verbose=1,
    )
    
    lazy_detector = LazyStrategyDetector(check_freq=10_000)
    progress_reporter = ProgressReporter(
        target_steps=model.num_timesteps + remaining_timesteps,
        report_freq=5_000
    )
    env_metrics_logger = EnvironmentMetricsLogger(log_freq=10_000)
    training_logger = TrainingLogger(
        log_file=os.path.join(log_dir, "training_log.json"),
        log_freq=5_000,
        flush_every=10_000,
    )
    _training_logger = training_logger

    callbacks = CallbackList([
        checkpoint_cb,
        lazy_detector,
        progress_reporter,
        env_metrics_logger,
        training_logger,
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # 6. TRAINING DEVAM
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n🚀 Eğitim devam ediyor...")
    print(f"   Kalan: {remaining_timesteps:,} timesteps")
    print(f"   Checkpoint: Her 20,000")
    print(f"   Ctrl+C: Güvenli kayıt (interrupted_model)")
    print("=" * 70 + "\n")

    model.learn(
        total_timesteps=remaining_timesteps,
        callback=callbacks,
        reset_num_timesteps=False,  # ← Sayacı sıfırlama!
        progress_bar=True,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 7. FINAL SAVE
    # ═══════════════════════════════════════════════════════════════════════
    model.save(os.path.join(models_dir, "final_model_1M"))
    env.save(os.path.join(models_dir, "vec_normalize_1M.pkl"))
    training_logger._flush()

    print("\n" + "=" * 70)
    print("✅ 1M timestep tamamlandı!")
    print(f"   Model: {models_dir}/final_model_1M.zip")
    print("=" * 70)

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()