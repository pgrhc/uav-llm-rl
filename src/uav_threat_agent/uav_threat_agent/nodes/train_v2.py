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

_model = None
_env = None
_models_dir = None
_training_logger = None


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



class LearningQualityMonitor(BaseCallback):
    def __init__(self, check_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
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
            except (AttributeError, KeyError):
                approx_kl = 0.0
                clip_frac = 0.0
                policy_loss = 0.0
                value_loss = 0.0

            log_entry = {
                "timesteps": int(self.num_timesteps),
                "time": datetime.now().isoformat(),
                "ep_rew_mean": ep_rew_mean,
                "ep_len_mean": ep_len_mean,
                "approx_kl": approx_kl,
                "clip_fraction": clip_frac,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
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
        except Exception as e:
            print(f"⚠️  JSON log hatası: {e}")

class CurriculumScheduler(BaseCallback):
    def __init__(self, stage1_end: int, stage2_end: int, verbose: int = 1):
        super().__init__(verbose)
        self.stage1_end = int(stage1_end)
        self.stage2_end = int(stage2_end)
        self._last_stage = None

    def _set_stage(self, stage: int):
        env0 = self.training_env.envs[0]
        real_env = getattr(env0, "unwrapped", env0)

        if hasattr(real_env, "set_curriculum_stage"):
            real_env.set_curriculum_stage(stage)
        else:
            print("⚠️ Env'de set_curriculum_stage yok!")

    def _on_training_start(self) -> None:
        self._set_stage(1)
        self._last_stage = 1
        print("🎓 Curriculum Stage = 1 (Person only)")

    def _on_step(self) -> bool:
        t = int(self.num_timesteps)

        if t < self.stage1_end:
            stage = 1
        elif t < self.stage2_end:
            stage = 2
        else:
            stage = 3

        if stage != self._last_stage:
            self._set_stage(stage)
            self._last_stage = stage
            print(f"🎓 Curriculum Stage = {stage}")

        return True


def main(args=None):
    global _model, _env, _models_dir, _training_logger
    
    if not rclpy.ok():
        rclpy.init(args=args)

    print("=" * 70)
    print("🚀 THREAT AGENT EĞİTİMİ V2 (MENTOR OPTIMIZED)")
    print("=" * 70)
    signal.signal(signal.SIGINT, signal_handler)
    SEED = 42
    set_random_seed(SEED)
    np.random.seed(SEED)
    run_name = f"PPO-V2-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    models_dir = f"models/{run_name}"
    log_dir = f"logs/{run_name}"
    _models_dir = models_dir
    
    for d in [models_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

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

    meta = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "seed": SEED,
        "env_id": "ThreatAgent-v11",
        "env_version": "V2",  
        "algo": "PPO",
        "hyperparameters": HYPERPARAMS,
        "vec_normalize": VEC_NORMALIZE_PARAMS,
        "improvements": [
            "Reduced n_steps: 4096 → 2048",
            "Reduced batch_size: 1024 → 256",
            "Reduced gamma: 0.995 → 0.99",
            "Added entropy: 0.01",
            "Better quality monitoring (MAE-based)",
            "Total timesteps: 204,800",
        ]
    }
    with open(os.path.join(log_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    def make_env():
        env = gym.make("ThreatAgent-v12")
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        try:
            env.reset(seed=SEED)
        except TypeError:
            pass
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, **VEC_NORMALIZE_PARAMS)
    _env = env

    model = PPO(
        "MlpPolicy", 
        env,
        device="cuda",
        verbose=1,
        tensorboard_log=log_dir,
        seed=SEED,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
        **HYPERPARAMS
    )
    _model = model
    TOTAL_TIMESTEPS = 204_800

    checkpoint_cb = CheckpointCallback(
        save_freq=4096, 
        save_path=models_dir,
        name_prefix="ppo_threat",
        save_vecnormalize=True,
        verbose=1,
    )
    quality_monitor = LearningQualityMonitor(check_freq=4096)
    
    progress_reporter = ProgressReporter(target_steps=TOTAL_TIMESTEPS, report_freq=4096)
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

    print(f"\n📊 Eğitim Ayarları (OPTIMIZED V2):")
    print(f"  Run Name:       {run_name}")
    print(f"  Total Steps:    {TOTAL_TIMESTEPS:,}")
    print(f"  Learning Rate:  {HYPERPARAMS['learning_rate']}")
    print(f"  N Steps:        {HYPERPARAMS['n_steps']} (↓ from 4096)")
    print(f"  Batch Size:     {HYPERPARAMS['batch_size']} (↓ from 1024)")
    print(f"  Gamma:          {HYPERPARAMS['gamma']} (↓ from 0.995)")
    print(f"  Entropy Coef:   {HYPERPARAMS['ent_coef']}")
    print(f"  Seed:           {SEED}")
    print(f"\n  Quality Check:  MAE-based (every 10k steps)")
    print(f"  Critical Alarm: MAE > 0.35 or Critical Miss > 50%")
    print(f"\n  TensorBoard:    tensorboard --logdir={log_dir}")
    print(f"  Ctrl+C:         Güvenli kayıt (interrupted_model)")
    print("=" * 70 + "\n")
    print("🚀 EĞİTİM BAŞLIYOR (204,800 timesteps, continuous monitoring)\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )
    
    model.save(os.path.join(models_dir, "final_model"))
    env.save(os.path.join(models_dir, "vec_normalize.pkl"))
    training_logger._flush()

    print("\n" + "=" * 70)
    print("✅ Eğitim tamamlandı!")
    print(f"   Model: {models_dir}/final_model.zip")
    print(f"   VecNorm: {models_dir}/vec_normalize.pkl")
    print(f"   Logs:  {log_dir}/training_log.json")
    print(f"   Metadata: {log_dir}/run_meta.json")
    print(f"   Monitor: {log_dir}/monitor.csv")
    print(f"   TensorBoard: tensorboard --logdir={log_dir}")
    print("=" * 70)

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()