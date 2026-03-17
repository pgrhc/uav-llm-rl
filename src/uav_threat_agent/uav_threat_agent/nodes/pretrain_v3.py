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

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from uav_threat_agent.nodes.asymmetric_policy import AsymmetricSACPolicy


_model = None
_env = None
_models_dir = None
_training_logger = None



def _extract_step_from_name(path: str) -> int:
    m = re.search(r"(\d+)_steps\.zip$", os.path.basename(path))
    if m:
        return int(m.group(1))
    base = os.path.basename(path)
    if "final_model" in base:
        return 10**12
    if "interrupted_model" in base:
        return 10**11
    return 0


def find_best_model_zip(resume_dir: str) -> str:
    p1 = os.path.join(resume_dir, "interrupted_model.zip")
    p2 = os.path.join(resume_dir, "final_model.zip")

    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2

    cand = glob.glob(os.path.join(resume_dir, "*.zip"))
    if not cand:
        raise FileNotFoundError(f"Model .zip bulunamadı: {resume_dir}")

    cand = sorted(cand, key=_extract_step_from_name, reverse=True)
    return cand[0]


def find_best_vecnorm(resume_dir: str) -> str:
    p1 = os.path.join(resume_dir, "vec_normalize_interrupted.pkl")
    p2 = os.path.join(resume_dir, "vec_normalize.pkl")

    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2


    cand = glob.glob(os.path.join(resume_dir, "*vecnormalize*.pkl"))
    if cand:
        cand = sorted(cand, reverse=True)
        return cand[0]

    raise FileNotFoundError(f"VecNormalize .pkl bulunamadı: {resume_dir}")


def get_real_env_from_vecnormalize(training_env):
    env = training_env
    while hasattr(env, "venv"):
        env = env.venv
    if hasattr(env, "envs") and len(env.envs) > 0:
        e0 = env.envs[0]
        return getattr(e0, "unwrapped", e0)
    return env


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
        print(f"⚠️ Kayıt hatası: {e}")
    finally:
        if _env is not None:
            _env.close()
        if rclpy.ok():
            rclpy.shutdown()
        print("\n👋 Güvenli şekilde kapatıldı.")
        sys.exit(0)


class LearningQualityMonitor(BaseCallback):
    def __init__(self, check_freq: int = 4096, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = int(check_freq)
        self.scores_buffer = []
        self.targets_buffer = []
        self.critical_objects = []

    def _safe_scalar(self, x, default=0.0):
        try:
            arr = np.asarray(x, dtype=np.float32)
            if arr.size == 0:
                return float(default)
            return float(arr.reshape(-1)[0])
        except Exception:
            return float(default)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        for info in infos:
            threats = info.get("top_threats", [])
            if not isinstance(threats, list):
                continue

            for threat in threats:
                if not isinstance(threat, dict):
                    continue

                score = self._safe_scalar(threat.get("score", 0.0))
                target = self._safe_scalar(threat.get("TRGT", 0.0))
                class_id = str(threat.get("cls", ""))
                dist = self._safe_scalar(threat.get("dist", 999.0))

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

        scores = np.asarray(self.scores_buffer, dtype=np.float32)
        targets = np.asarray(self.targets_buffer, dtype=np.float32)

        mae = float(np.mean(np.abs(scores - targets)))
        if np.std(scores) > 0.01 and np.std(targets) > 0.01:
            correlation = float(np.corrcoef(scores, targets)[0, 1])
        else:
            correlation = 0.0

        if len(self.critical_objects) > 0:
            critical_misses = sum(1 for s, _, _, _ in self.critical_objects if s < 0.5)
            critical_miss_rate = float(critical_misses / len(self.critical_objects))
        else:
            critical_miss_rate = 0.0

        zero_ratio = float(np.mean(scores < 0.1))
        high_ratio = float(np.mean(scores > 0.5))
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
            warnings.append("🚨 CRITICAL: MAE > 0.35")
        elif mae > 0.25:
            warnings.append("⚠️ WARNING: MAE > 0.25")

        if critical_miss_rate > 0.5:
            warnings.append("🚨 CRITICAL: Critical miss > 50%")
        elif critical_miss_rate > 0.3:
            warnings.append("⚠️ WARNING: Critical miss > 30%")

        if correlation < 0.1:
            warnings.append("⚠️ WARNING: Low correlation")
        if score_std < 0.05:
            warnings.append("⚠️ WARNING: Std < 0.05")

        if warnings:
            print("\n🚨 ALARMLAR:")
            for w in warnings:
                print(f"  {w}")
        else:
            print("\n✅ Öğrenme kalitesi sağlıklı görünüyor")

        print("=" * 70 + "\n")


class ProgressReporter(BaseCallback):
    def __init__(self, target_steps: int, report_freq: int = 4096):
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

        print(f"\n⏱️ {self.num_timesteps:,} / {self.target_steps:,} timesteps")
        print(f"   FPS: {fps:.1f} | Kalan: ~{hours}h {mins}m")
        self.last_report_time = now


class EnvironmentMetricsLogger(BaseCallback):
    def __init__(self, log_freq: int = 4096):
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
                    cls = str(threat.get("cls", ""))
                    score = float(np.asarray(threat.get("score", 0.0)).reshape(-1)[0])

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
    def __init__(self, log_file: str, log_freq: int = 4096, flush_every: int = 8192):
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
                nl = self.model.logger.name_to_value
                actor_loss = float(nl.get("train/actor_loss", 0.0))
                critic_loss = float(nl.get("train/critic_loss", 0.0))
                ent_coef = float(nl.get("train/ent_coef", 0.0))
                ent_coef_loss = float(nl.get("train/ent_coef_loss", 0.0))
            except Exception:
                actor_loss = critic_loss = ent_coef = ent_coef_loss = 0.0

            self.logs.append({
                "timesteps": int(self.num_timesteps),
                "time": datetime.now().isoformat(),
                "ep_rew_mean": ep_rew_mean,
                "ep_len_mean": ep_len_mean,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "ent_coef": ent_coef,
                "ent_coef_loss": ent_coef_loss,
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
            print(f"⚠️ JSON log hatası: {e}")


class CurriculumScheduler(BaseCallback):
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
            print("⚠️ Env'de set_curriculum_stage yok!")

    def _stage_for_t(self, t: int) -> int:
        if t < self.stage1_end:
            return 1
        elif t < self.stage2_end:
            return 2
        else:
            return 3

    def _on_training_start(self) -> None:
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
    parser.add_argument("--resume_dir", required=True, help="Eski SAC eğitim klasörü")
    parser.add_argument("--env_id", default="ThreatAgent-v13")
    parser.add_argument("--total_timesteps", type=int, default=307_200, help="Resume sonrası hedef toplam timestep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Finetune LR")
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

    resume_dir = args.resume_dir

    run_name = f"SAC-AsymAC-Resume-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    models_dir = f"models/{run_name}"
    log_dir = f"logs/{run_name}"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    vec_path = find_best_vecnorm(resume_dir)
    model_path = find_best_model_zip(resume_dir)

    _models_dir = models_dir


    HYPERPARAMS = {
        "learning_rate":   args.learning_rate,
        "buffer_size":     50_000,
        "learning_starts": 10_000,
        "batch_size":      256,
        "tau":             0.005,
        "gamma":           0.99,
        "train_freq":      1,
        "gradient_steps":  1,
        "ent_coef":        0.01,
        "use_sde":         False,
    }

    VEC_NORMALIZE_PARAMS = {
        "norm_obs": True,
        "norm_reward": False,
        "clip_obs": 10.0,
        "gamma": HYPERPARAMS["gamma"],
    }

    def make_env():
        env = gym.make(args.env_id)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        try:
            env.reset(seed=SEED)
        except TypeError:
            pass
        return env

    base_env = DummyVecEnv([make_env])

    vec_path = find_best_vecnorm(resume_dir)
    env = VecNormalize.load(vec_path, base_env)
    env.training = True
    env.norm_reward = VEC_NORMALIZE_PARAMS["norm_reward"]
    env.clip_obs = VEC_NORMALIZE_PARAMS["clip_obs"]
    _env = env

    model_path = find_best_model_zip(resume_dir)

    print("=" * 70)
    print("🔁 RESUME / FINETUNE TRAINING (SAC Asymmetric)")
    print("=" * 70)
    print(f"  Env:         {args.env_id}")
    print(f"  Model .zip:  {model_path}")
    print(f"  VecNorm .pkl:{vec_path}")
    print(f"  Target total:{args.total_timesteps:,}")
    print(f"  LR:          {args.learning_rate}")
    print("=" * 70)

    model = SAC.load(
        model_path,
        env=env,
        device="cuda",
        tensorboard_log=log_dir,
        seed=SEED,
        print_system_info=False,
        custom_objects={
            "learning_rate": HYPERPARAMS["learning_rate"],
            "buffer_size": HYPERPARAMS["buffer_size"],
            "learning_starts": HYPERPARAMS["learning_starts"],
            "batch_size": HYPERPARAMS["batch_size"],
            "tau": HYPERPARAMS["tau"],
            "gamma": HYPERPARAMS["gamma"],
            "train_freq": HYPERPARAMS["train_freq"],
            "gradient_steps": HYPERPARAMS["gradient_steps"],
            "ent_coef": HYPERPARAMS["ent_coef"],
            "use_sde": HYPERPARAMS["use_sde"],
            "policy_class": AsymmetricSACPolicy,
        }
    )
    _model = model

    checkpoint_cb = CheckpointCallback(
        save_freq=4096,
        save_path=models_dir,
        name_prefix="sac_asymac_resume",
        save_vecnormalize=True,
        verbose=1,
    )

    quality_monitor = LearningQualityMonitor(check_freq=4096)
    progress_reporter = ProgressReporter(target_steps=args.total_timesteps, report_freq=4096)
    env_metrics_logger = EnvironmentMetricsLogger(log_freq=4096)

    training_logger = TrainingLogger(
        log_file=os.path.join(log_dir, "training_log_resume.json"),
        log_freq=4096,
        flush_every=8192,
    )
    _training_logger = training_logger

    curriculum_cb = CurriculumScheduler(stage1_end=40_960, stage2_end=102_400)

    callbacks = CallbackList([
        checkpoint_cb,
        quality_monitor,
        progress_reporter,
        env_metrics_logger,
        training_logger,
        curriculum_cb,
    ])

    print(f"🔁 Resume hedef toplam step: {args.total_timesteps:,}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        log_interval=2,
        reset_num_timesteps=False,
    )

    model.save(os.path.join(models_dir, "final_model"))
    env.save(os.path.join(models_dir, "vec_normalize.pkl"))
    training_logger._flush()

    print("\n" + "=" * 70)
    print("✅ Resume / finetune eğitim tamamlandı!")
    print(f"   Model:   {models_dir}/final_model.zip")
    print(f"   VecNorm: {models_dir}/vec_normalize.pkl")
    print(f"   Logs:    {log_dir}/training_log_resume.json")
    print("=" * 70)

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()