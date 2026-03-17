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
from rclpy.node import Node
from std_msgs.msg import Int32
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

TOPIC_SET_STAGE = "/curriculum/set_stage"   

_ros_node: "TrainerROSNode | None" = None


class TrainerROSNode(Node):
    """
    Sadece /curriculum/set_stage topic'ine publish eder.
    maze_curriculum.py bu komutu alarak Gazebo'yu yönetir.
    """

    def __init__(self):
        super().__init__("trainer_curriculum_publisher")
        self._pub = self.create_publisher(Int32, TOPIC_SET_STAGE, 10)
        self.get_logger().info(f"TrainerROSNode hazir — PUB {TOPIC_SET_STAGE}")

    def publish_stage(self, stage: int):
        msg = Int32()
        msg.data = stage
        self._pub.publish(msg)
        self.get_logger().info(f"Stage {stage} publish edildi -> {TOPIC_SET_STAGE}")



_model            = None
_env              = None
_models_dir       = None
_training_logger  = None


def signal_handler(sig, frame):
    print("\n\nCtrl+C algilandi! Model kaydediliyor...")
    try:
        if _model is not None and _models_dir is not None:
            save_path = os.path.join(_models_dir, "interrupted_model")
            _model.save(save_path)
            print(f"Model kaydedildi: {save_path}.zip")

        if _env is not None and _models_dir is not None:
            vec_path = os.path.join(_models_dir, "vec_normalize_interrupted.pkl")
            _env.save(vec_path)
            print(f"VecNormalize kaydedildi: {vec_path}")

        if _training_logger is not None:
            _training_logger._flush()
            print("Training log kaydedildi")

    except Exception as e:
        print(f"Kayit hatasi: {e}")
    finally:
        if _env is not None:
            _env.close()
        if rclpy.ok():
            rclpy.shutdown()
        print("Guvenli sekilde kapatildi.")
        sys.exit(0)


class LearningQualityMonitor(BaseCallback):
    def __init__(self, check_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq     = check_freq
        self.scores_buffer  = []
        self.targets_buffer = []
        self.critical_objects = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True
        for info in infos:
            for threat in info.get("top_threats", []):
                score    = float(threat.get("score", 0.0))
                target   = float(threat.get("TRGT",  0.0))
                class_id = threat.get("cls", "")
                dist     = float(threat.get("dist",  999.0))
                self.scores_buffer.append(score)
                self.targets_buffer.append(target)
                if class_id in ["Person", "Unknown"] and dist < 5.0 and target > 0.5:
                    self.critical_objects.append((score, target, class_id, dist))

        if self.num_timesteps % self.check_freq == 0:
            self._analyze_and_report()
            self.scores_buffer    = []
            self.targets_buffer   = []
            self.critical_objects = []
        return True

    def _analyze_and_report(self):
        if not self.scores_buffer:
            return
        scores  = np.array(self.scores_buffer)
        targets = np.array(self.targets_buffer)
        mae     = float(np.mean(np.abs(scores - targets)))
        corr    = (float(np.corrcoef(scores, targets)[0, 1])
                   if np.std(scores) > 0.01 and np.std(targets) > 0.01 else 0.0)
        crit_miss = (sum(1 for s, t, _, _ in self.critical_objects if s < 0.5)
                     / len(self.critical_objects) if self.critical_objects else 0.0)
        zero_r  = float(np.sum(scores < 0.1) / scores.size)
        high_r  = float(np.sum(scores > 0.5) / scores.size)
        std_s   = float(np.std(scores))

        self.logger.record("quality/mae",                mae)
        self.logger.record("quality/correlation",        corr)
        self.logger.record("quality/critical_miss_rate", crit_miss)
        self.logger.record("quality/score_std",          std_s)
        self.logger.record("quality/zero_ratio",         zero_r)
        self.logger.record("quality/high_ratio",         high_r)

        print(f"\n{'='*70}")
        print(f"LEARNING QUALITY CHECK @ {self.num_timesteps:,} timesteps")
        print(f"{'='*70}")
        print(f"  MAE:              {mae:.3f}")
        print(f"  Correlation:      {corr:+.3f}")
        print(f"  Critical Miss:    {crit_miss:.1%}")
        print(f"  Score Std:        {std_s:.3f}")
        print(f"  Zero / High Ratio:{zero_r:.1%} / {high_r:.1%}")

        warnings = []
        if mae > 0.35:        warnings.append("CRITICAL: MAE > 0.35")
        elif mae > 0.25:      warnings.append("WARNING:  MAE > 0.25")
        if crit_miss > 0.5:   warnings.append("CRITICAL: Critical miss > 50%")
        elif crit_miss > 0.3: warnings.append("WARNING:  Critical miss > 30%")
        if corr < 0.1:        warnings.append("WARNING:  Low correlation")
        if std_s < 0.05:      warnings.append("WARNING:  Std < 0.05 (collapsed)")
        for w in warnings: print(f"  {w}")
        if not warnings: print("  Ogrenme kalitesi saglikli")
        print(f"{'='*70}\n")


class ProgressReporter(BaseCallback):
    def __init__(self, target_steps: int, report_freq: int = 5_000):
        super().__init__()
        self.target_steps = int(target_steps)
        self.report_freq  = int(report_freq)
        self.start_time   = time.time()
        self.last_time    = self.start_time

    def _on_step(self) -> bool:
        if self.num_timesteps % self.report_freq == 0:
            now      = time.time()
            elapsed  = now - self.start_time
            interval = now - self.last_time
            fps      = self.report_freq / interval if interval > 0 else 0.0
            rem_sec  = (max(0, self.target_steps - self.num_timesteps) * elapsed
                        / self.num_timesteps) if self.num_timesteps > 0 else 0.0
            h, m     = int(rem_sec // 3600), int((rem_sec % 3600) // 60)
            print(f"\n{self.num_timesteps:,} / {self.target_steps:,} | FPS: {fps:.1f} | Kalan: ~{h}h {m}m")
            self.last_time = now
        return True


class EnvironmentMetricsLogger(BaseCallback):
    def __init__(self, log_freq: int = 10_000):
        super().__init__()
        self.log_freq         = int(log_freq)
        self.person_count     = 0
        self.high_score_count = 0
        self.total_objects    = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos:
            for info in infos:
                for threat in info.get("top_threats") or []:
                    self.total_objects += 1
                    if threat.get("cls") == "Person": self.person_count += 1
                    if float(threat.get("score", 0.0)) > 0.5: self.high_score_count += 1

        if self.num_timesteps % self.log_freq == 0 and self.total_objects > 0:
            p_r = self.person_count     / self.total_objects
            h_r = self.high_score_count / self.total_objects
            self.logger.record("env/person_detection_ratio", p_r)
            self.logger.record("env/high_score_ratio",       h_r)
            print(f"\nENV METRICS @ {self.num_timesteps:,}  | Person: {p_r:.1%}  | HighScore: {h_r:.1%}")
            self.person_count = self.high_score_count = self.total_objects = 0
        return True


class TrainingLogger(BaseCallback):
    def __init__(self, log_file: str, log_freq: int = 5_000, flush_every: int = 10_000):
        super().__init__()
        self.log_file   = log_file
        self.log_freq   = int(log_freq)
        self.flush_every= int(flush_every)
        self.logs       = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            ep_buf = list(getattr(self.model, "ep_info_buffer", []))
            entry  = {
                "timesteps":   int(self.num_timesteps),
                "time":        datetime.now().isoformat(),
                "ep_rew_mean": float(np.mean([e.get("r", 0.0) for e in ep_buf])) if ep_buf else 0.0,
                "ep_len_mean": float(np.mean([e.get("l", 0.0) for e in ep_buf])) if ep_buf else 0.0,
            }
            try:
                nl = self.model.logger.name_to_value
                entry.update({
                    "approx_kl":   float(nl.get("train/approx_kl", 0.0)),
                    "clip_frac":   float(nl.get("train/clip_fraction", 0.0)),
                    "policy_loss": float(nl.get("train/policy_gradient_loss", 0.0)),
                    "value_loss":  float(nl.get("train/value_loss", 0.0)),
                })
            except (AttributeError, KeyError):
                pass
            self.logs.append(entry)
            if self.num_timesteps % self.flush_every == 0:
                self._flush()
        return True

    def _flush(self):
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.logs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"JSON log hatasi: {e}")



class CurriculumScheduler(BaseCallback):
    """
    Timestep sinirlarina gore stage hesaplar ve
    /curriculum/set_stage topic'ine publish eder.
    maze_curriculum.py bu komutu alarak:
      Stage 1 -> sadece statik aktorler
      Stage 2 -> statikler silinir, dinamik aktorler
      Stage 3 -> dinamikler kalir + statikler eklenir
    """

    def __init__(self, stage1_end: int, stage2_end: int, verbose: int = 1):
        super().__init__(verbose)
        self.stage1_end  = int(stage1_end)
        self.stage2_end  = int(stage2_end)
        self._last_stage = None

    def _publish_stage(self, stage: int):
        global _ros_node
        if _ros_node is not None:
            _ros_node.publish_stage(stage)
        else:
            print(f"WARNING: _ros_node None, stage {stage} publish edilemedi!")

        try:
            env0     = self.training_env.envs[0]
            real_env = getattr(env0, "unwrapped", env0)
            if hasattr(real_env, "set_curriculum_stage"):
                real_env.set_curriculum_stage(stage)
        except Exception:
            pass

    def _on_training_start(self) -> None:
        self._publish_stage(1)
        self._last_stage = 1
        print(f"\nCurriculum Stage = 1  (0 - {self.stage1_end:,} timesteps)  -> STATIC only")

    def _on_step(self) -> bool:
        t = int(self.num_timesteps)

        if t < self.stage1_end:
            stage = 1
        elif t < self.stage2_end:
            stage = 2
        else:
            stage = 3

        if stage != self._last_stage:
            self._publish_stage(stage)
            self._last_stage = stage

            desc = {
                1: "STATIC only",
                2: "DYNAMIC only  (statics removed)",
                3: "MIXED  (static + dynamic)",
            }.get(stage, "?")
            print(f"\nCurriculum Stage = {stage}  [{t:,} timesteps]  -> {desc}")

        return True


def main(args=None):
    global _model, _env, _models_dir, _training_logger, _ros_node

    if not rclpy.ok():
        rclpy.init(args=args)


    _ros_node = TrainerROSNode()
    import threading
    ros_thread = threading.Thread(target=rclpy.spin, args=(_ros_node,), daemon=True)
    ros_thread.start()

    print("=" * 70)
    print("THREAT AGENT EGITIMI V2")
    print("=" * 70)
    signal.signal(signal.SIGINT, signal_handler)

    SEED = 42
    set_random_seed(SEED)
    np.random.seed(SEED)

    run_name   = f"PPO-V2-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    models_dir = f"models/{run_name}"
    log_dir    = f"logs/{run_name}"
    _models_dir= models_dir
    for d in [models_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    HYPERPARAMS = {
        "learning_rate": 3e-4,
        "n_steps":       2048,
        "batch_size":    256,
        "n_epochs":      10,
        "gamma":         0.99,
        "gae_lambda":    0.95,
        "clip_range":    0.2,
        "ent_coef":      0.01,
        "vf_coef":       0.5,
        "max_grad_norm": 0.5,
    }

    VEC_NORMALIZE_PARAMS = {
        "norm_obs":    True,
        "norm_reward": False,
        "clip_obs":    10.0,
        "clip_reward": 5.0,
        "gamma":       HYPERPARAMS["gamma"],
    }

   
    TOTAL_TIMESTEPS = 204_800
    STAGE1_END      = 70_000   
    STAGE2_END      = 140_000  
                             

    meta = {
        "run_name":       run_name,
        "created_at":     datetime.now().isoformat(),
        "seed":           SEED,
        "env_id":         "ThreatAgent-v12",
        "algo":           "PPO",
        "curriculum": {
            "stage1_end": STAGE1_END,
            "stage2_end": STAGE2_END,
            "total":      TOTAL_TIMESTEPS,
            "ros_topic":  TOPIC_SET_STAGE,
        },
        "hyperparameters": HYPERPARAMS,
        "vec_normalize":   VEC_NORMALIZE_PARAMS,
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

    env  = DummyVecEnv([make_env])
    env  = VecNormalize(env, **VEC_NORMALIZE_PARAMS)
    _env = env

    model = PPO(
        "MlpPolicy", env,
        device="cuda", verbose=1,
        tensorboard_log=log_dir, seed=SEED,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
        **HYPERPARAMS,
    )
    _model = model


    training_logger   = TrainingLogger(
        log_file    = os.path.join(log_dir, "training_log.json"),
        log_freq    = 5_000,
        flush_every = 20_000,
    )
    _training_logger = training_logger

   
    curriculum_cb = CurriculumScheduler(
        stage1_end = STAGE1_END,
        stage2_end = STAGE2_END,
    )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq        = 4096,
            save_path        = models_dir,
            name_prefix      = "ppo_threat",
            save_vecnormalize= True,
            verbose          = 1,
        ),
        LearningQualityMonitor(check_freq=4096),
        ProgressReporter(target_steps=TOTAL_TIMESTEPS, report_freq=4096),
        EnvironmentMetricsLogger(log_freq=4096),
        training_logger,
        curriculum_cb,      
    ])

    print(f"\nEgitim Ayarlari:")
    print(f"  Run Name:       {run_name}")
    print(f"  Total Steps:    {TOTAL_TIMESTEPS:,}")
    print(f"  Stage 1 (static only):   0          -> {STAGE1_END:,}")
    print(f"  Stage 2 (dynamic only):  {STAGE1_END:,} -> {STAGE2_END:,}")
    print(f"  Stage 3 (mixed):         {STAGE2_END:,} -> {TOTAL_TIMESTEPS:,}")
    print(f"\n  Stage komutu topic:  {TOPIC_SET_STAGE}")
    print(f"  maze_curriculum.py bu topic'i dinlemeli!\n")
    print(f"  TensorBoard: tensorboard --logdir={log_dir}")
    print("=" * 70 + "\n")

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = callbacks,
        progress_bar    = True,
    )

    model.save(os.path.join(models_dir, "final_model"))
    env.save(os.path.join(models_dir, "vec_normalize.pkl"))
    training_logger._flush()

    print("\n" + "=" * 70)
    print("Egitim tamamlandi!")
    print(f"  Model:    {models_dir}/final_model.zip")
    print(f"  VecNorm:  {models_dir}/vec_normalize.pkl")
    print(f"  Logs:     {log_dir}/training_log.json")
    print(f"  TensorBoard: tensorboard --logdir={log_dir}")
    print("=" * 70)

    env.close()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()