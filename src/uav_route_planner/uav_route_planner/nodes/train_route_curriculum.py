#!/usr/bin/env python3
"""
train_route_curriculum.py — 3-stage curriculum training for Route Agent

Stages:
    1  (0 - 150K)   Path following — open maze, no actors
    2  (150K - 350K) Static obstacles — narrow corridors, dead-ends
    3  (350K - 600K) Dynamic threats — 180-200 walking actors

Callbacks:
    CurriculumScheduler     Stage transition + lazy actor spawning
    RouteTrainingMonitor    Policy entropy, action std, value loss,
                            success/collision/timeout rates, episode length,
                            deterministic-vs-stochastic comparison,
                            route quality metrics (path error, threat exposure)
    TrajectoryRecorder      Per-episode position + entropy/std for heatmap
    ProgressReporter        ETA and FPS

Usage:
    ros2 run uav_route_planner train_route_curriculum
"""

import os
import sys
import time
import json
import signal
import math
from datetime import datetime
from collections import deque

import numpy as np
import gymnasium as gym
import rclpy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

import uav_route_planner.envs  # noqa: F401  — triggers register()

from uav_route_planner.maze_curriculum_world import spawn_stage3_actors_lazy

from std_msgs.msg import Int32

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBALS FOR SIGNAL HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

_model = None
_env = None
_save_dir = None
_trajectory_recorder = None


def _signal_handler(sig, frame):
    print("\n\nCtrl+C algilandi! Model kaydediliyor...")
    try:
        if _model is not None and _save_dir is not None:
            path = os.path.join(_save_dir, "interrupted_model")
            _model.save(path)
            print(f"Model kaydedildi: {path}.zip")

        if _env is not None and _save_dir is not None:
            vn_path = os.path.join(_save_dir, "vec_normalize_interrupted.pkl")
            _env.save(vn_path)
            print(f"VecNormalize kaydedildi: {vn_path}")

        if _trajectory_recorder is not None:
            _trajectory_recorder.flush()
            print("Trajectory data kaydedildi")

    except Exception as e:
        print(f"Kayit hatasi: {e}")
    finally:
        if _env is not None:
            _env.close()
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK 1: CURRICULUM SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

class CurriculumScheduler(BaseCallback):
    """Manages stage transitions and lazy actor spawning."""

    STAGE_RANGES = {
        1: (0, 150_000),
        2: (150_000, 350_000),
        3: (350_000, 600_000),
    }

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.current_stage = 1
        self._stage_pub = None
        self._actors_spawned = False

    def _on_training_start(self):
        raw_env = self._get_raw_env()
        if raw_env and hasattr(raw_env, "node"):
            self._stage_pub = raw_env.node.create_publisher(
                Int32, "/route/set_stage", 10
            )

    def _on_step(self) -> bool:
        target_stage = self._stage_for_timestep(self.num_timesteps)
        if target_stage != self.current_stage:
            self._transition(target_stage)
        return True

    def _stage_for_timestep(self, ts):
        for stage, (start, end) in self.STAGE_RANGES.items():
            if start <= ts < end:
                return stage
        return 3

    def _transition(self, stage):
        raw_env = self._get_raw_env()
        if raw_env is not None:
            raw_env.set_curriculum_stage(stage)

        if self._stage_pub is not None:
            msg = Int32()
            msg.data = stage
            self._stage_pub.publish(msg)

        if stage == 3 and not self._actors_spawned:
            print("\nStage 3: Aktorler spawn ediliyor...")
            try:
                spawn_stage3_actors_lazy()
                self._actors_spawned = True
                print("Aktorler spawn edildi.")
            except Exception as e:
                print(f"Aktor spawn hatasi: {e}")

        self.current_stage = stage
        self.logger.record("curriculum/stage", stage)

        print(f"\n{'=' * 60}")
        print(f"STAGE {stage} BASLADI @ {self.num_timesteps:,} timesteps")
        print(f"{'=' * 60}\n")

    def _get_raw_env(self):
        try:
            return self.training_env.venv.envs[0].unwrapped
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK 2: ROUTE TRAINING MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class RouteTrainingMonitor(BaseCallback):
    """
    Unified monitoring callback:
    - Policy entropy, action std, value loss
    - Success / collision / timeout rates
    - Episode length, mean reward
    - Deterministic vs stochastic action comparison
    - Route quality: path error, threat exposure
    """

    def __init__(self, log_freq=2048, compare_freq=10_000, window=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.compare_freq = compare_freq
        self.window = window

        self.ep_successes = deque(maxlen=window)
        self.ep_collisions = deque(maxlen=window)
        self.ep_timeouts = deque(maxlen=window)
        self.ep_lengths = deque(maxlen=window)
        self.ep_rewards = deque(maxlen=window)
        self.ep_path_errors = deque(maxlen=window)
        self.ep_threat_maxes = deque(maxlen=window)

        self._step_path_errors = []
        self._step_threat_maxes = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            self._step_path_errors.append(info.get("path_error", 0.0))
            self._step_threat_maxes.append(info.get("max_threat", 0.0))

            if dones[i]:
                self.ep_successes.append(1.0 if info.get("success") else 0.0)
                self.ep_collisions.append(1.0 if info.get("collision") else 0.0)
                self.ep_timeouts.append(1.0 if info.get("timeout") else 0.0)
                ep_info = info.get("episode")
                if ep_info:
                    self.ep_lengths.append(ep_info.get("l", 0))
                    self.ep_rewards.append(ep_info.get("r", 0.0))

                if self._step_path_errors:
                    self.ep_path_errors.append(np.mean(self._step_path_errors))
                if self._step_threat_maxes:
                    self.ep_threat_maxes.append(np.mean(self._step_threat_maxes))

                self._step_path_errors.clear()
                self._step_threat_maxes.clear()

        if self.num_timesteps % self.log_freq == 0:
            self._log_training_metrics()
            self._log_episode_metrics()

        if self.num_timesteps % self.compare_freq == 0:
            self._log_det_vs_stoch()

        return True

    def _log_training_metrics(self):
        vals = getattr(self.model.logger, "name_to_value", {})

        entropy = vals.get("train/entropy_loss", None)
        if entropy is not None:
            self.logger.record("monitor/policy_entropy", entropy)

        value_loss = vals.get("train/value_loss", None)
        if value_loss is not None:
            self.logger.record("monitor/value_loss", value_loss)

        approx_kl = vals.get("train/approx_kl", None)
        if approx_kl is not None:
            self.logger.record("monitor/approx_kl", approx_kl)

        clip_frac = vals.get("train/clip_fraction", None)
        if clip_frac is not None:
            self.logger.record("monitor/clip_fraction", clip_frac)

        try:
            log_std = self.model.policy.log_std.data.cpu().numpy()
            action_std = float(np.mean(np.exp(log_std)))
            self.logger.record("monitor/action_std_mean", action_std)
            self.logger.record("monitor/action_log_std_mean", float(np.mean(log_std)))
        except (AttributeError, RuntimeError):
            pass

    def _log_episode_metrics(self):
        if len(self.ep_successes) > 0:
            self.logger.record("episode/success_rate", np.mean(self.ep_successes))
        if len(self.ep_collisions) > 0:
            self.logger.record("episode/collision_rate", np.mean(self.ep_collisions))
        if len(self.ep_timeouts) > 0:
            self.logger.record("episode/timeout_rate", np.mean(self.ep_timeouts))
        if len(self.ep_lengths) > 0:
            self.logger.record("episode/mean_length", np.mean(self.ep_lengths))
        if len(self.ep_rewards) > 0:
            self.logger.record("episode/mean_reward", np.mean(self.ep_rewards))
        if len(self.ep_path_errors) > 0:
            self.logger.record("route/mean_path_error", np.mean(self.ep_path_errors))
        if len(self.ep_threat_maxes) > 0:
            self.logger.record("route/mean_threat_exposure",
                               np.mean(self.ep_threat_maxes))

    def _log_det_vs_stoch(self):
        try:
            new_obs = self.locals.get("new_obs")
            if new_obs is None:
                return

            det_action, _ = self.model.predict(new_obs, deterministic=True)
            stoch_action = self.locals.get("actions")
            if stoch_action is None:
                return

            diff = float(np.mean(np.abs(det_action - stoch_action)))
            self.logger.record("monitor/det_stoch_action_diff", diff)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK 3: TRAJECTORY RECORDER
# ═══════════════════════════════════════════════════════════════════════════════

class TrajectoryRecorder(BaseCallback):
    """Records per-episode drone trajectories with confidence data for heatmaps."""

    def __init__(self, save_dir, max_episodes=500, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.max_episodes = max_episodes

        self._current_positions = []
        self._current_rewards = []
        self._current_action_stds = []
        self._episodes = []
        self._episode_count = 0

    def _on_step(self) -> bool:
        raw_env = self._get_raw_env()
        if raw_env is None:
            return True

        self._current_positions.append([
            raw_env.drone_x, raw_env.drone_y, raw_env.drone_z
        ])

        reward = self.locals.get("rewards", [0.0])
        self._current_rewards.append(float(reward[0]) if len(reward) > 0 else 0.0)

        try:
            log_std = self.model.policy.log_std.data.cpu().numpy()
            mean_std = float(np.mean(np.exp(log_std)))
        except (AttributeError, RuntimeError):
            mean_std = 1.0
        self._current_action_stds.append(mean_std)

        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        if len(dones) > 0 and dones[0]:
            info = infos[0] if infos else {}
            outcome = "success" if info.get("success") else \
                      "collision" if info.get("collision") else \
                      "timeout" if info.get("timeout") else "unknown"

            episode_data = {
                "episode": self._episode_count,
                "stage": info.get("stage", 1),
                "timestep": self.num_timesteps,
                "positions": self._current_positions,
                "rewards": self._current_rewards,
                "action_stds": self._current_action_stds,
                "total_reward": sum(self._current_rewards),
                "length": len(self._current_positions),
                "outcome": outcome,
            }
            self._episodes.append(episode_data)
            self._episode_count += 1

            self._current_positions = []
            self._current_rewards = []
            self._current_action_stds = []

            if len(self._episodes) >= self.max_episodes:
                self.flush()

        return True

    def flush(self):
        if not self._episodes:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.save_dir, f"trajectories_{ts}.json")
        try:
            with open(path, "w") as f:
                json.dump(self._episodes, f, indent=1)
            print(f"Trajectory data kaydedildi: {path} ({len(self._episodes)} ep)")
        except Exception as e:
            print(f"Trajectory kayit hatasi: {e}")
        self._episodes = []

    def _get_raw_env(self):
        try:
            return self.training_env.venv.envs[0].unwrapped
        except Exception:
            return None

    def _on_training_end(self):
        self.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK 4: PROGRESS REPORTER
# ═══════════════════════════════════════════════════════════════════════════════

class ProgressReporter(BaseCallback):
    def __init__(self, total_timesteps, report_freq=10_000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.report_freq = report_freq
        self.start_time = None
        self.last_report_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        self.last_report_time = self.start_time

    def _on_step(self) -> bool:
        if self.num_timesteps % self.report_freq == 0:
            self._report()
        return True

    def _report(self):
        now = time.time()
        elapsed_total = now - self.start_time
        elapsed_interval = now - self.last_report_time

        fps = self.report_freq / max(elapsed_interval, 1e-6)
        pct = 100.0 * self.num_timesteps / self.total_timesteps

        remaining = max(0, self.total_timesteps - self.num_timesteps)
        if self.num_timesteps > 0:
            eta_sec = remaining * (elapsed_total / self.num_timesteps)
            h, m = int(eta_sec // 3600), int((eta_sec % 3600) // 60)
        else:
            h, m = 0, 0

        print(
            f"\n[{pct:5.1f}%] {self.num_timesteps:,}/{self.total_timesteps:,} "
            f"| FPS: {fps:.0f} | ETA: {h}h {m}m"
        )
        self.last_report_time = now


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOG CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingLogWriter(BaseCallback):
    """Writes periodic training stats to a JSON file."""

    def __init__(self, log_file, log_freq=5000):
        super().__init__()
        self.log_file = log_file
        self.log_freq = log_freq
        self.entries = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        vals = getattr(self.model.logger, "name_to_value", {})
        ep_buf = list(self.model.ep_info_buffer) if hasattr(self.model, "ep_info_buffer") else []

        entry = {
            "timesteps": self.num_timesteps,
            "time": datetime.now().isoformat(),
            "ep_rew_mean": float(np.mean([e["r"] for e in ep_buf])) if ep_buf else 0.0,
            "ep_len_mean": float(np.mean([e["l"] for e in ep_buf])) if ep_buf else 0.0,
            "entropy": vals.get("train/entropy_loss", None),
            "value_loss": vals.get("train/value_loss", None),
            "approx_kl": vals.get("train/approx_kl", None),
        }
        self.entries.append(entry)

        if len(self.entries) % 10 == 0:
            self._flush()

        return True

    def _flush(self):
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "w") as f:
                json.dump(self.entries, f, indent=2)
        except Exception as e:
            print(f"JSON log yazma hatasi: {e}")

    def _on_training_end(self):
        self._flush()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

TOTAL_TIMESTEPS = 600_000


def main(args=None):
    global _model, _env, _save_dir, _trajectory_recorder

    if not rclpy.ok():
        rclpy.init(args=args)

    signal.signal(signal.SIGINT, _signal_handler)

    run_name = f"RouteCurriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    models_dir = os.path.join("models", run_name)
    log_dir = os.path.join("logs", run_name)
    traj_dir = os.path.join(log_dir, "trajectories")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)
    _save_dir = models_dir

    print("=" * 60)
    print("ROUTE AGENT CURRICULUM TRAINING")
    print("=" * 60)
    print(f"Run:     {run_name}")
    print(f"Models:  {models_dir}")
    print(f"Logs:    {log_dir}")
    print(f"Total:   {TOTAL_TIMESTEPS:,} timesteps")
    print(f"Stages:  1 (0-150K) | 2 (150K-350K) | 3 (350K-600K)")
    print("=" * 60)

    # --- Environment ---
    def make_env():
        env = gym.make("RouteCurriculumAgent-v0")
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        return env

    vec_env = DummyVecEnv([make_env])
    env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    _env = env

    # --- Model ---
    from uav_route_planner.networks.route_extractor import RouteCombinedExtractor

    policy_kwargs = dict(
        features_extractor_class=RouteCombinedExtractor,
        net_arch=[256, 128],
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.3,
        ent_coef=0.05,
        policy_kwargs=policy_kwargs,
    )
    _model = model

    # --- Callbacks ---
    curriculum_scheduler = CurriculumScheduler(verbose=1)

    route_monitor = RouteTrainingMonitor(
        log_freq=2048, compare_freq=10_000, window=100,
    )

    trajectory_recorder = TrajectoryRecorder(
        save_dir=traj_dir, max_episodes=200,
    )
    _trajectory_recorder = trajectory_recorder

    progress_reporter = ProgressReporter(
        total_timesteps=TOTAL_TIMESTEPS, report_freq=10_000,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=models_dir,
        name_prefix="route_curriculum",
        save_vecnormalize=True,
        verbose=1,
    )

    training_log = TrainingLogWriter(
        log_file=os.path.join(log_dir, "training_log.json"),
        log_freq=5_000,
    )

    callbacks = CallbackList([
        curriculum_scheduler,
        route_monitor,
        trajectory_recorder,
        progress_reporter,
        checkpoint_cb,
        training_log,
    ])

    # --- Train ---
    print("\nEgitim basliyor...\n")

    SAVE_INTERVAL = 50_000
    steps_done = 0

    while steps_done < TOTAL_TIMESTEPS:
        chunk = min(SAVE_INTERVAL, TOTAL_TIMESTEPS - steps_done)
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            tb_log_name="RouteCurriculum",
            callback=callbacks,
        )
        steps_done += chunk

        model.save(os.path.join(models_dir, f"route_curriculum_{steps_done}"))
        env.save(os.path.join(models_dir, f"vecnorm_{steps_done}.pkl"))
        print(f"\nCheckpoint kaydedildi: {steps_done:,}/{TOTAL_TIMESTEPS:,}")

    # --- Final save ---
    model.save(os.path.join(models_dir, "final_model"))
    env.save(os.path.join(models_dir, "vec_normalize_final.pkl"))
    trajectory_recorder.flush()

    print("\n" + "=" * 60)
    print("EGITIM TAMAMLANDI")
    print(f"Model: {models_dir}/final_model.zip")
    print("=" * 60)

    env.close()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
