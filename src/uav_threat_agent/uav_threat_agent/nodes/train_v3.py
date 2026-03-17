#!/usr/bin/env python3
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import uav_threat_agent   
from uav_threat_agent.nodes.asymmetric_policy import AsymmetricSACPolicy

TOPIC_SET_STAGE = "/curriculum/set_stage"
_ros_node: "TrainerROSNode | None" = None


class TrainerROSNode(Node):
    def __init__(self):
        super().__init__("trainer_curriculum_publisher")
        self._pub = self.create_publisher(Int32, TOPIC_SET_STAGE, 10)
        self.get_logger().info(f"TrainerROSNode hazir – PUB {TOPIC_SET_STAGE}")

    def publish_stage(self, stage: int) -> None:
        msg = Int32(); msg.data = stage
        self._pub.publish(msg)


_model           = None
_env             = None
_models_dir      = None
_training_logger = None


def signal_handler(sig, frame):
    print("\nCtrl+C – kaydediliyor...")
    try:
        if _model and _models_dir:
            _model.save(os.path.join(_models_dir, "interrupted_model"))
        if _env and _models_dir:
            _env.save(os.path.join(_models_dir, "vec_normalize_interrupted.pkl"))
        if _training_logger:
            _training_logger._flush()
    except Exception as e:
        print(f"Kayit hatasi: {e}")
    finally:
        if _env:       _env.close()
        if rclpy.ok(): rclpy.shutdown()
        sys.exit(0)


class LearningQualityMonitor(BaseCallback):
    def __init__(self, check_freq: int = 2048):
        super().__init__()
        self.freq        = check_freq
        self.scores_buf  = []
        self.targets_buf = []
        self.crit_buf    = []

    def _on_step(self) -> bool:
        for info in (self.locals.get("infos") or []):
            for t in info.get("top_threats", []):
                s = float(t.get("score", 0.0))
                g = float(t.get("TRGT",  0.0))
                self.scores_buf.append(s)
                self.targets_buf.append(g)
                if t.get("cls") in ["Person", "Unknown"] and \
                   float(t.get("dist", 999)) < 5.0 and g > 0.5:
                    self.crit_buf.append((s, g))

        if self.num_timesteps % self.freq == 0 and self.scores_buf:
            s  = np.array(self.scores_buf)
            g  = np.array(self.targets_buf)
            mae  = float(np.mean(np.abs(s - g)))
            corr = float(np.corrcoef(s, g)[0, 1]) \
                   if np.std(s) > 0.01 and np.std(g) > 0.01 else 0.0
            cmiss = (sum(1 for sc, _ in self.crit_buf if sc < 0.5)
                     / len(self.crit_buf)) if self.crit_buf else 0.0
            std_s = float(np.std(s))

            self.logger.record("quality/mae",                mae)
            self.logger.record("quality/correlation",        corr)
            self.logger.record("quality/critical_miss_rate", cmiss)
            self.logger.record("quality/score_std",          std_s)
            self.logger.record("quality/zero_ratio",  float(np.mean(s < 0.1)))
            self.logger.record("quality/high_ratio",  float(np.mean(s > 0.5)))

            print(f"\n{'='*60}")
            print(f"QUALITY @ {self.num_timesteps:,}")
            print(f"  MAE={mae:.3f}  Corr={corr:+.3f}  "
                  f"CritMiss={cmiss:.1%}  Std={std_s:.3f}")
            warns = []
            if mae   > 0.35: warns.append("CRITICAL: MAE > 0.35")
            if cmiss > 0.50: warns.append("CRITICAL: CritMiss > 50%")
            if corr  < 0.10: warns.append("WARNING:  Dusuk korelasyon")
            if std_s < 0.05: warns.append("WARNING:  Std < 0.05")
            for w in warns: print(f"  {w}")
            if not warns: print("  Saglikli")
            print(f"{'='*60}")
            self.scores_buf  = []
            self.targets_buf = []
            self.crit_buf    = []
        return True


class ProgressReporter(BaseCallback):
    def __init__(self, target: int, freq: int = 2048):
        super().__init__()
        self.target = int(target)
        self.freq   = int(freq)
        self.t0     = time.time()
        self.last   = self.t0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.freq == 0:
            now   = time.time()
            fps   = self.freq / max(now - self.last, 1e-6)
            rem   = (max(0, self.target - self.num_timesteps)
                     * (now - self.t0) / max(self.num_timesteps, 1))
            h, m  = int(rem // 3600), int((rem % 3600) // 60)
            print(f"\n{self.num_timesteps:,}/{self.target:,} | FPS:{fps:.0f} | ~{h}h{m}m")
            self.last = now
        return True


class TrainingLogger(BaseCallback):
    def __init__(self, log_file: str, log_freq: int = 2048,
                 flush_every: int = 4096):
        super().__init__()
        self.log_file    = log_file
        self.log_freq    = log_freq
        self.flush_every = flush_every
        self.logs: list  = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True
        buf   = list(getattr(self.model, "ep_info_buffer", []))
        entry = {
            "timesteps":   self.num_timesteps,
            "time":        datetime.now().isoformat(),
            "ep_rew_mean": float(np.mean([e.get("r", 0) for e in buf])) if buf else 0.0,
            "ep_len_mean": float(np.mean([e.get("l", 0) for e in buf])) if buf else 0.0,
        }
        try:
            nl = self.model.logger.name_to_value
            for k, v in [("actor_loss",  "train/actor_loss"),
                         ("critic_loss", "train/critic_loss"),
                         ("ent_coef",    "train/ent_coef"),
                         ("ent_loss",    "train/ent_coef_loss")]:
                entry[k] = float(nl.get(v, 0.0))
        except Exception:
            pass
        self.logs.append(entry)
        if self.num_timesteps % self.flush_every == 0:
            self._flush()
        return True

    def _flush(self):
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "w") as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            print(f"Log hatasi: {e}")


class CurriculumScheduler(BaseCallback):
    STAGE_DESC = {
        1: "STATIC only  – Person",
        2: "DYNAMIC added – Person + Unknown",
        3: "MIXED         – all classes",
    }

    def __init__(self, stage1_end: int, stage2_end: int):
        super().__init__()
        self.stage1_end  = int(stage1_end)
        self.stage2_end  = int(stage2_end)
        self._last_stage = None

    def _push(self, stage: int) -> None:
        global _ros_node
        if _ros_node:
            _ros_node.publish_stage(stage)
        try:
            env0 = self.training_env.envs[0]
            real = getattr(env0, "unwrapped", env0)
            if hasattr(real, "set_curriculum_stage"):
                real.set_curriculum_stage(stage)
        except Exception:
            pass

    def _on_training_start(self) -> None:
        self._push(1)
        self._last_stage = 1
        print(f"\nCurriculum Stage=1  {self.STAGE_DESC[1]}")
        print(f"  Stage 1:  0 – {self.stage1_end:,}")
        print(f"  Stage 2:  {self.stage1_end:,} – {self.stage2_end:,}")
        print(f"  Stage 3:  {self.stage2_end:,} – son")

    def _on_step(self) -> bool:
        t     = self.num_timesteps
        stage = 1 if t < self.stage1_end else 2 if t < self.stage2_end else 3
        if stage != self._last_stage:
            self._push(stage)
            self._last_stage = stage
            print(f"\nCurriculum Stage={stage}  [{t:,}]  {self.STAGE_DESC[stage]}")
        return True

def main(args=None):
    global _model, _env, _models_dir, _training_logger, _ros_node

    if not rclpy.ok():
        rclpy.init(args=args)

    _ros_node  = TrainerROSNode()
    ros_thread = threading.Thread(
        target=rclpy.spin, args=(_ros_node,), daemon=True
    )
    ros_thread.start()

    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 65)
    print("SAC + ASİMETRİK AKTÖR-KRİTİK EGİTİMİ")
    print("=" * 65)

    SEED = 42
    set_random_seed(SEED)
    np.random.seed(SEED)

    run_name    = f"SAC-AsymAC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    models_dir  = f"models/{run_name}"
    log_dir     = f"logs/{run_name}"
    _models_dir = models_dir
    for d in [models_dir, log_dir]:
        os.makedirs(d, exist_ok=True)


    TOTAL_TIMESTEPS = 204_800   
    STAGE1_END      = 40_960
    STAGE2_END      = 102_400

    HYPERPARAMS = {
        "learning_rate":   3e-4,
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

    VEC_NORM = {
        "norm_obs":    True,
        "norm_reward": False,          
        "clip_obs":    10.0,
        "gamma":       HYPERPARAMS["gamma"],
    }

    def make_env():
        env = gym.make("ThreatAgent-v13")
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        try:
            env.reset(seed=SEED)
        except TypeError:
            pass
        return env

    env  = DummyVecEnv([make_env])
    env  = VecNormalize(env, **VEC_NORM)
    _env = env

    model = SAC(
        AsymmetricSACPolicy,
        env,
        device          = "cuda",
        verbose         = 1,
        tensorboard_log = log_dir,
        seed            = SEED,
        policy_kwargs   = dict(
            pi_arch = [256, 256, 128],   
            vf_arch = [256, 256, 128],   
        ),
        **HYPERPARAMS,
    )
    _model = model


    meta = {
        "run_name":   run_name,
        "created_at": datetime.now().isoformat(),
        "algo":       "SAC",
        "policy":     "AsymmetricSACPolicy",
        "obs_dim":    74,
        "priv_dim":   43,
        "critic_input": "obs(74) + priv(43) + action(5) = 122",
        "priv_components": {
            "[0:5]":   "target_scores    (5)  — reward fn gercek risk",
            "[5:15]":  "actor_token_xy   (10) — token-hizali [x,y]x5",
            "[15:23]": "drone_global     (8)  — x,y,z,sin_yaw,cos_yaw,vx,vy,vz",
            "[23:33]": "maze_feat        (10) — cell_face_dist(4)+cell_norm(2)+walls_cur(4)",
            "[33:38]": "waypoint_rel     (5)  — dx,dy,dist,sin_angle,cos_angle",
            "[38:43]": "prev_action      (5)",
        },
        "curriculum": {
            "stage1": f"0–{STAGE1_END:,}  static (Person only)",
            "stage2": f"{STAGE1_END:,}–{STAGE2_END:,}  dynamic (Person+Unknown)",
            "stage3": f"{STAGE2_END:,}–{TOTAL_TIMESTEPS:,}  mixed",
        },
        "hyperparameters": {k: str(v) for k, v in HYPERPARAMS.items()},
    }
    with open(os.path.join(log_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    training_logger = TrainingLogger(
        log_file    = os.path.join(log_dir, "training_log.json"),
        log_freq    = 2048,
        flush_every = 4096,
    )
    _training_logger = training_logger

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq         = 4096, 
            save_path         = models_dir,
            name_prefix       = "sac_asymac",
            save_vecnormalize = True,
        ),
        LearningQualityMonitor(check_freq=2048),
        ProgressReporter(target=TOTAL_TIMESTEPS, freq=2048),
        training_logger,
        CurriculumScheduler(stage1_end=STAGE1_END, stage2_end=STAGE2_END),
    ])

    print(f"\n  Algo:      SAC (off-policy, entropy regularized)")
    print(f"  Actor:     obs(74) → MLP(256,256,128) → action(5)")
    print(f"  Critic:    obs(74)+priv(43)+action(5)=122 → MLP(256,256,128) → Q")
    print(f"\n  Priv 43-dim bileşimi:")
    for k, v in meta["priv_components"].items():
        print(f"    {k}  {v}")
    print(f"\n  Stage 1:   0 – {STAGE1_END:,}    static Person")
    print(f"  Stage 2:   {STAGE1_END:,} – {STAGE2_END:,}   dynamic +Unknown")
    print(f"  Stage 3:   {STAGE2_END:,} – {TOTAL_TIMESTEPS:,}   mixed all")
    print(f"\n  TensorBoard: tensorboard --logdir={log_dir}")
    print("=" * 65 + "\n")

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = callbacks,
        progress_bar    = True,
        log_interval    = 2,
    )

    model.save(os.path.join(models_dir, "final_model"))
    env.save(os.path.join(models_dir, "vec_normalize.pkl"))
    training_logger._flush()

    print(f"\nTamamlandi → {models_dir}/final_model.zip")
    env.close()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()