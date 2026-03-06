#!/usr/bin/env python3
import os
import re
import glob
import time
import json
import signal
import sys
import argparse
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


_model = None
_env = None
_models_dir = None
_training_logger = None


# -----------------------------
# Utils: find latest checkpoint
# -----------------------------
def _extract_step_from_name(path: str) -> int:
    # matches "..._123456_steps.zip" or similar
    m = re.search(r"(\d+)_steps\.zip$", os.path.basename(path))
    if m:
        return int(m.group(1))
    # fallback: interrupted/final -> treat as very large priority
    base = os.path.basename(path)
    if "final_model" in base:
        return 10**12
    if "interrupted_model" in base:
        return 10**11
    return 0


def find_best_model_zip(resume_dir: str) -> str:
    # Priority: interrupted_model.zip, final_model.zip, else latest checkpoint
    cand = []
    p1 = os.path.join(resume_dir, "interrupted_model.zip")
    p2 = os.path.join(resume_dir, "final_model.zip")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2

    cand.extend(glob.glob(os.path.join(resume_dir, "*.zip")))
    if not cand:
        raise FileNotFoundError(f"Model .zip bulunamadı: {resume_dir}")

    cand = sorted(cand, key=_extract_step_from_name, reverse=True)
    return cand[0]


def find_best_vecnorm(resume_dir: str) -> str:
    # Priority: vec_normalize_interrupted.pkl, vec_normalize.pkl
    p1 = os.path.join(resume_dir, "vec_normalize_interrupted.pkl")
    p2 = os.path.join(resume_dir, "vec_normalize.pkl")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"VecNormalize .pkl bulunamadı: {resume_dir}")


def get_real_env_from_vecnormalize(training_env):
    """
    Callback içinde gerçek env'e ulaşmak için güvenli unwrap.
    training_env genelde VecNormalize (VecEnvWrapper) olur.
    """
    env = training_env
    # unwrap VecNormalize -> DummyVecEnv
    while hasattr(env, "venv"):
        env = env.venv
    # DummyVecEnv -> envs[0]
    if hasattr(env, "envs") and len(env.envs) > 0:
        e0 = env.envs[0]
        return getattr(e0, "unwrapped", e0)
    return env


# -----------------------------
# Ctrl+C safe shutdown
# -----------------------------
def signal_handler(sig, frame):
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
            print("✅ Training log kaydedildi")

    except Exception as e:
        print(f"⚠️  Kayıt hatası: {e}")
    finally:
        if _env is not None:
            _env.close()
        if rclpy.ok():
            rclpy.shutdown()
        print("\n👋 Güvenli şekilde kapatıldı.")
        sys.exit(0)


# -----------------------------
# Callbacks (seninle aynı)
# -----------------------------
class LearningQualityMonitor(BaseCallback):
    def __init__(self, check_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = int(check_freq)
        self.scores_buffer = []
        self.targets_buffer = []
        self.critical_objects = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        for info in infos:
            threats = info.get("top_threats", [])
            for threat in threats:
                score = float(threat.get("score", 0.0))
                target = float(threat.get("TRGT", 0.0))
                class_id = threat.get("cls", "")
                dist = float(threat.get("dist", 999.0))

                self.scores_buffer.append(score)
                self.targets_buffer.append(target)

                is_critical = (class_id in ["Person", "Unknown"]) and (dist < 5.0) and (target > 0.5)
                if is_critical:
                    self.critical_objects.append((score, target, class_id, dist))

        if self.num_timesteps % self.check_freq == 0:
            self._analyze_and_report()
            self.scores_buffer = []
            self.targets_buffer = []
            self.critical_objects = []

        return True

    def _analyze_and_report(self):
        if len(self.scores_buffer) == 0:
            return

        scores = np.array(self.scores_buffer)
        targets = np.array(self.targets_buffer)

        mae = float(np.mean(np.abs(scores - targets)))
        if np.std(scores) > 0.01 and np.std(targets) > 0.01:
            correlation = float(np.corrcoef(scores, targets)[0, 1])
        else:
            correlation = 0.0

        if len(self.critical_objects) > 0:
            critical_misses = sum(1 for s, t, _, _ in self.critical_objects if s < 0.5)
            critical_miss_rate = float(critical_misses / len(self.critical_objects))
        else:
            critical_miss_rate = 0.0

        zero_ratio = float(np.sum(scores < 0.1) / scores.size)
        high_ratio = float(np.sum(scores > 0.5) / scores.size)
        score_std = float(np.std(scores))

        self.logger.record("quality/mae", mae)
        self.logger.record("quality/correlation", correlation)
        self.logger.record("quality/critical_miss_rate", critical_miss_rate)
        self.logger.record("quality/score_std", score_std)
        self.logger.record("quality/zero_ratio", zero_ratio)
        self.logger.record("quality/high_ratio", high_ratio)

        print("\n" + "=" * 70)
        print(f"🎯 LEARNING QUALITY CHECK @ {self.num_timesteps:,} timesteps")
        print("=" * 70)
        print(f"  MAE (Score-Target):     {mae:.3f}")
        print(f"  Correlation:            {correlation:+.3f}")
        print(f"  Critical Miss Rate:     {critical_miss_rate:.1%}")
        print(f"  Score Std:              {score_std:.3f}")
        print(f"  Zero Ratio:             {zero_ratio:.1%}")
        print(f"  High Ratio:             {high_ratio:.1%}")

        warnings = []
        if mae > 0.35:
            warnings.append("🚨 CRITICAL: MAE > 0.35 → Ajan hedefi anlamıyor!")
        elif mae > 0.25:
            warnings.append("⚠️  WARNING: MAE > 0.25 → Öğrenme yavaş")

        if critical_miss_rate > 0.5:
            warnings.append("🚨 CRITICAL: Critical miss > 50% → Yakın tehditler görülmüyor!")
        elif critical_miss_rate > 0.3:
            warnings.append("⚠️  WARNING: Critical miss > 30%")

        if correlation < 0.1:
            warnings.append("⚠️  WARNING: Low correlation → Skorlar rastgele")

        if score_std < 0.05:
            warnings.append("⚠️  WARNING: Std < 0.05 → Hep aynı değer veriyor")

        if warnings:
            print("\n🚨 ALARMLAR:")
            for w in warnings:
                print(f"  {w}")
        else:
            print("\n✅ Öğrenme kalitesi sağlıklı görünüyor")

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

            try:
                approx_kl = float(self.model.logger.name_to_value.get("train/approx_kl", 0.0))
                clip_frac = float(self.model.logger.name_to_value.get("train/clip_fraction", 0.0))
                policy_loss = float(self.model.logger.name_to_value.get("train/policy_gradient_loss", 0.0))
                value_loss = float(self.model.logger.name_to_value.get("train/value_loss", 0.0))
            except Exception:
                approx_kl = clip_frac = policy_loss = value_loss = 0.0

            self.logs.append({
                "timesteps": int(self.num_timesteps),
                "time": datetime.now().isoformat(),
                "ep_rew_mean": ep_rew_mean,
                "ep_len_mean": ep_len_mean,
                "approx_kl": approx_kl,
                "clip_fraction": clip_frac,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
            })

            if self.num_timesteps % self.flush_every == 0:
                self._flush()

        return True

    def _flush(self):
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.logs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  JSON log hatası: {e}")


class CurriculumScheduler(BaseCallback):
    """
    Resume uyumlu:
    - Eğitim yeniden başlasa bile timesteps'e göre stage seçer.
    - Ayrıca başlangıçta 'mevcut timesteps' üzerinden stage set eder.
    """
    def __init__(self, stage1_end: int, stage2_end: int, verbose: int = 1):
        super().__init__(verbose)
        self.stage1_end = int(stage1_end)
        self.stage2_end = int(stage2_end)
        self._last_stage = None

    def _set_stage(self, stage: int):
        real_env = get_real_env_from_vecnormalize(self.training_env)
        if hasattr(real_env, "set_curriculum_stage"):
            real_env.set_curriculum_stage(stage)
        else:
            print("⚠️ Env'de set_curriculum_stage yok! (Curriculum çalışmayacak)")

    def _stage_for_t(self, t: int) -> int:
        if t < self.stage1_end:
            return 1
        elif t < self.stage2_end:
            return 2
        else:
            return 3

    def _on_training_start(self) -> None:
        # resume olsa bile mevcut num_timesteps'e göre stage ayarla
        stage = self._stage_for_t(int(self.num_timesteps))
        self._set_stage(stage)
        self._last_stage = stage
        print(f"🎓 Curriculum Stage = {stage} (start)")

    def _on_step(self) -> bool:
        stage = self._stage_for_t(int(self.num_timesteps))
        if stage != self._last_stage:
            self._set_stage(stage)
            self._last_stage = stage
            print(f"🎓 Curriculum Stage = {stage}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_dir", required=True, help="Eski eğitim klasörü (models/... içinde .zip ve vecnormalize .pkl olmalı)")
    parser.add_argument("--env_id", default="ThreatAgent-v12")
    parser.add_argument("--total_timesteps", type=int, default=102_400, help="Toplam hedef timesteps (resume dahil)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global _model, _env, _models_dir, _training_logger

    if not rclpy.ok():
        rclpy.init()

    signal.signal(signal.SIGINT, signal_handler)

    SEED = args.seed
    set_random_seed(SEED)
    np.random.seed(SEED)

    resume_dir = args.resume_dir
    if not os.path.isdir(resume_dir):
        raise FileNotFoundError(f"resume_dir klasör değil: {resume_dir}")

    # Aynı klasöre kaydetmeye devam edelim
    models_dir = resume_dir
    log_dir = resume_dir.replace("models/", "logs/", 1) if resume_dir.startswith("models/") else os.path.join(resume_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    _models_dir = models_dir

    # Hiperparametreler aynı kalmalı (resume güvenliği)
    HYPERPARAMS = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    VEC_NORMALIZE_PARAMS = {
        "norm_obs": True,
        "norm_reward": False,
        "clip_obs": 10.0,
        "clip_reward": 5.0,
        "gamma": HYPERPARAMS["gamma"],
    }

    # --- make env ---
    def make_env():
        env = gym.make(args.env_id)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        try:
            env.reset(seed=SEED)
        except TypeError:
            pass
        return env

    base_env = DummyVecEnv([make_env])

    # --- load VecNormalize ---
    vec_path = find_best_vecnorm(models_dir)
    env = VecNormalize.load(vec_path, base_env)
    env.training = True
    env.norm_reward = VEC_NORMALIZE_PARAMS["norm_reward"]
    env.clip_obs = VEC_NORMALIZE_PARAMS["clip_obs"]
    env.clip_reward = VEC_NORMALIZE_PARAMS["clip_reward"]
    _env = env

    # --- load model ---
    model_path = find_best_model_zip(models_dir)
    print("=" * 70)
    print("🔁 RESUME TRAINING")
    print("=" * 70)
    print(f"  Env:         {args.env_id}")
    print(f"  Model .zip:   {model_path}")
    print(f"  VecNorm .pkl: {vec_path}")
    print(f"  Target total: {args.total_timesteps:,}")
    print("=" * 70)

    model = PPO.load(
        model_path,
        env=env,
        device="cuda",
        tensorboard_log=log_dir,
        seed=SEED,
        print_system_info=False,
    )
    _model = model

    # --- callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=4096,
        save_path=models_dir,
        name_prefix="ppo_threat",
        save_vecnormalize=True,
        verbose=1,
    )

    quality_monitor = LearningQualityMonitor(check_freq=4096)
    progress_reporter = ProgressReporter(target_steps=args.total_timesteps, report_freq=4096)
    env_metrics_logger = EnvironmentMetricsLogger(log_freq=4096)

    training_logger = TrainingLogger(
        log_file=os.path.join(log_dir, "training_log.json"),
        log_freq=5_000,
        flush_every=20_000,
    )
    _training_logger = training_logger

    curriculum_cb = CurriculumScheduler(stage1_end=70_000, stage2_end=140_000)

    callbacks = CallbackList([
        checkpoint_cb,
        quality_monitor,
        progress_reporter,
        env_metrics_logger,
        training_logger,
        curriculum_cb,
    ])

    
    target_steps =  args.total_timesteps

    print(f"🔁 Resume step: {target_steps}")
    print(f"🎯 Yeni hedef: {target_steps}")
    # IMPORTANT: resume için reset_num_timesteps=False
    model.learn(
        total_timesteps=target_steps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    # Final save (aynı klasöre)
    model.save(os.path.join(models_dir, "final_model"))
    env.save(os.path.join(models_dir, "vec_normalize.pkl"))
    training_logger._flush()

    print("\n" + "=" * 70)
    print("✅ Resume eğitim tamamlandı!")
    print(f"   Model:   {models_dir}/final_model.zip")
    print(f"   VecNorm: {models_dir}/vec_normalize.pkl")
    print(f"   Logs:    {log_dir}/training_log.json")
    print("=" * 70)

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()