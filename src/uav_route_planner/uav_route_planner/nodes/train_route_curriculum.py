#!/usr/bin/env python3
import os
import sys
import time
import json
import signal
import math
import atexit
import threading
from datetime import datetime
from collections import deque

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
import gymnasium as gym
import rclpy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import glob

import uav_route_planner.envs
UNIFIED_MAZE = False


def spawn_stage3_actors_lazy():
    try:
        from uav_route_planner.maze_curriculum_world import spawn_stage3_actors_lazy as _fn
        return _fn()
    except ModuleNotFoundError:
        print(
            "WARNING: maze_curriculum_world yok. Stage 3 aktorleri spawn edilmedi."
        )
        return []


from std_msgs.msg import Int32


class TimestepSyncWrapper(gym.Wrapper):
    def set_sb3_timesteps(self, n: int) -> None:
        self.env.unwrapped.set_sb3_timesteps(int(n))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class SyncSB3TimestepCallback(BaseCallback):
    def _on_step(self) -> bool:
        try:
            self.training_env.env_method("set_sb3_timesteps", int(self.num_timesteps))
        except Exception:
            pass
        return True


def _vec_env_call_method(training_env, method_name: str, *args, indices=None):
    venv = training_env.venv
    idx = indices if indices is not None else [0]
    return venv.env_method(method_name, *args, indices=idx)[0]


_shutdown_lock = threading.Lock()
_shutdown_done = False
_model = None
_env = None
_save_dir = None
_trajectory_recorder = None


def _do_shutdown():
    global _shutdown_done
    with _shutdown_lock:
        if _shutdown_done:
            return
        _shutdown_done = True
        global _model, _env, _save_dir, _trajectory_recorder
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
                try:
                    _env.close()
                except (EOFError, BrokenPipeError, OSError, ValueError):
                    pass
                except Exception:
                    pass
            if rclpy.ok():
                rclpy.shutdown()
            _model = _env = _save_dir = _trajectory_recorder = None


def _signal_handler(sig, frame):
    print("\n\nCtrl+C algilandi! Model kaydediliyor...")
    _do_shutdown()
    sys.exit(0)


class CurriculumScheduler(BaseCallback):
    @staticmethod
    def default_stage_ranges():
        t = int(os.environ.get("ROUTE_TOTAL_TIMESTEPS", "1000000"))
        s1 = int(os.environ.get("ROUTE_STAGE1_END", "550000"))
        s2 = int(os.environ.get("ROUTE_STAGE2_END", "800000"))
        s1 = max(0, min(s1, t))
        s2 = max(s1, min(s2, t))
        return {1: (0, s1), 2: (s1, s2), 3: (s2, t)}

    def __init__(self, verbose=1, stage_pub=None):
        super().__init__(verbose)
        self.current_stage = 1
        self.stage_ranges = CurriculumScheduler.default_stage_ranges()
        self._stage_pub = stage_pub
        self._actors_spawned = False

    def _on_training_start(self):
        if UNIFIED_MAZE:
            return
        if self._stage_pub is not None:
            return
        raw_env = self._get_raw_env()
        if raw_env and hasattr(raw_env, "node"):
            self._stage_pub = raw_env.node.create_publisher(
                Int32, "/route/set_stage", 10
            )

    def _on_step(self) -> bool:
        if UNIFIED_MAZE:
            return True
        target_stage = self._stage_for_timestep(self.num_timesteps)
        if target_stage != self.current_stage:
            self._transition(target_stage)
        return True

    def _stage_for_timestep(self, ts):
        for stage, (start, end) in self.stage_ranges.items():
            if start <= ts < end:
                return stage
        return 3

    def _transition(self, stage):
        try:
            _vec_env_call_method(self.training_env, "set_curriculum_stage", stage)
        except Exception as e:
            print(f"[CurriculumScheduler] set_curriculum_stage: {e}")

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

        try:
            self.model.policy.optimizer.state.clear()
            print(f"[CurriculumScheduler] Adam optimizer momentum reset for Stage {stage}.")
        except Exception as e:
            print(f"[CurriculumScheduler] Optimizer reset failed (non-fatal): {e}")

        try:
            self.model.ep_info_buffer.clear()
        except Exception:
            pass

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


# ── Reward component keys (simplified) ──────────────────────────────────────
# These match the info dict keys populated in step() of the env.
# collision / success / timeout are terminal flags, not per-step scalars,
# so they are tracked separately as episode outcomes.


class RouteTrainingMonitor(BaseCallback):
    """Logs episode outcomes and key diagnostics to TensorBoard / SB3 logger."""

    def __init__(self, log_freq=2048, window=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.window = window
        self._ep_successes   = deque(maxlen=window)
        self._ep_collisions  = deque(maxlen=window)
        self._ep_timeouts    = deque(maxlen=window)
        self._ep_lengths     = deque(maxlen=window)
        self._ep_rewards     = deque(maxlen=window)
        self._ep_min_lidar   = deque(maxlen=window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            done_i = dones[i] if i < len(dones) else False
            if done_i:
                self._ep_successes.append(1.0 if info.get("success") else 0.0)
                self._ep_collisions.append(1.0 if info.get("collision") else 0.0)
                self._ep_timeouts.append(1.0 if info.get("timeout") else 0.0)
                ep = info.get("episode")
                if ep:
                    self._ep_lengths.append(ep.get("l", 0))
                    self._ep_rewards.append(ep.get("r", 0.0))
                self._ep_min_lidar.append(float(info.get("min_lidar_m", 30.0)))

        if self.num_timesteps % self.log_freq == 0:
            self._log()
        return True

    def _log(self):
        if self._ep_successes:
            self.logger.record("episode/success_rate",   np.mean(self._ep_successes))
        if self._ep_collisions:
            self.logger.record("episode/collision_rate", np.mean(self._ep_collisions))
        if self._ep_timeouts:
            self.logger.record("episode/timeout_rate",   np.mean(self._ep_timeouts))
        if self._ep_lengths:
            self.logger.record("episode/mean_length",    np.mean(self._ep_lengths))
        if self._ep_rewards:
            self.logger.record("episode/mean_reward",    np.mean(self._ep_rewards))
        if self._ep_min_lidar:
            self.logger.record("route/mean_min_lidar_m", np.mean(self._ep_min_lidar))

        vals = getattr(self.model.logger, "name_to_value", {})
        for src_key, dst_key in [
            ("train/entropy_loss",  "monitor/policy_entropy"),
            ("train/value_loss",    "monitor/value_loss"),
            ("train/approx_kl",     "monitor/approx_kl"),
            ("train/clip_fraction", "monitor/clip_fraction"),
        ]:
            v = vals.get(src_key)
            if v is not None:
                self.logger.record(dst_key, v)

        try:
            log_std = self.model.policy.log_std.data.cpu().numpy()
            self.logger.record("monitor/action_std_mean", float(np.mean(np.exp(log_std))))
        except (AttributeError, RuntimeError):
            pass


class TrajectoryRecorder(BaseCallback):
    def __init__(self, save_dir, max_episodes=500, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.max_episodes = max_episodes
        self._current_positions = []
        self._current_rewards   = []
        self._episodes          = []
        self._episode_count     = 0

    def _on_step(self) -> bool:
        try:
            pos = _vec_env_call_method(self.training_env, "get_drone_position")
        except Exception:
            return True
        if pos is None:
            return True
        arr = np.asarray(pos, dtype=np.float64).ravel()
        if arr.size < 3:
            return True
        self._current_positions.append([float(arr[0]), float(arr[1]), float(arr[2])])

        reward = self.locals.get("rewards", [0.0])
        self._current_rewards.append(float(reward[0]) if len(reward) > 0 else 0.0)

        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        if len(dones) > 0 and dones[0]:
            info    = infos[0] if infos else {}
            outcome = (
                "success"   if info.get("success")   else
                "collision" if info.get("collision") else
                "timeout"   if info.get("timeout")   else "unknown"
            )
            self._episodes.append({
                "episode":      self._episode_count,
                "stage":        info.get("stage", 1),
                "timestep":     self.num_timesteps,
                "positions":    self._current_positions,
                "rewards":      self._current_rewards,
                "total_reward": sum(self._current_rewards),
                "length":       len(self._current_positions),
                "outcome":      outcome,
            })
            self._episode_count      += 1
            self._current_positions  = []
            self._current_rewards    = []
            if len(self._episodes) >= self.max_episodes:
                self.flush()
        return True

    def flush(self):
        if not self._episodes:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.save_dir, f"trajectories_{ts}.json")
        try:
            with open(path, "w") as f:
                json.dump(self._episodes, f, indent=1)
            print(f"Trajectory data kaydedildi: {path} ({len(self._episodes)} ep)")
        except Exception as e:
            print(f"Trajectory kayit hatasi: {e}")
        self._episodes = []

    def _on_training_end(self):
        self.flush()


class ProgressReporter(BaseCallback):
    def __init__(self, total_timesteps, report_freq=10_000):
        super().__init__()
        self.total_timesteps  = total_timesteps
        self.report_freq      = report_freq
        self.start_time       = None
        self.last_report_time = None
        self.last_reported_step = 0

    def _on_training_start(self):
        self.start_time         = time.time()
        self.last_report_time   = self.start_time
        self.last_reported_step = 0
        print(f"[train] Rapor her {self.report_freq:,} adimda.", flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.last_reported_step + self.report_freq:
            self._report()
            self.last_reported_step = self.num_timesteps
        return True

    def _report(self):
        now              = time.time()
        elapsed_total    = now - self.start_time
        elapsed_interval = now - self.last_report_time
        fps              = self.report_freq / max(elapsed_interval, 1e-6)
        pct              = 100.0 * self.num_timesteps / self.total_timesteps
        remaining        = max(0, self.total_timesteps - self.num_timesteps)
        if self.num_timesteps > 0:
            eta_sec = remaining * (elapsed_total / self.num_timesteps)
            h, m    = int(eta_sec // 3600), int((eta_sec % 3600) // 60)
        else:
            h, m = 0, 0
        print(
            f"\n[{pct:5.1f}%] {self.num_timesteps:,}/{self.total_timesteps:,} "
            f"| FPS: {fps:.0f} | ETA: {h}h {m}m",
            flush=True,
        )
        self.last_report_time = now


class TrainingLogWriter(BaseCallback):
    def __init__(self, log_file, log_freq=5000):
        super().__init__()
        self.log_file = log_file
        self.log_freq = log_freq
        self.entries  = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True
        vals   = getattr(self.model.logger, "name_to_value", {})
        ep_buf = list(self.model.ep_info_buffer) if hasattr(self.model, "ep_info_buffer") else []
        entry  = {
            "timesteps":    self.num_timesteps,
            "time":         datetime.now().isoformat(),
            "ep_rew_mean":  float(np.mean([e["r"] for e in ep_buf])) if ep_buf else 0.0,
            "ep_len_mean":  float(np.mean([e["l"] for e in ep_buf])) if ep_buf else 0.0,
            "entropy":      vals.get("train/entropy_loss") or vals.get("train/entropy"),
            "value_loss":   vals.get("train/value_loss"),
            "approx_kl":    vals.get("train/approx_kl"),
            "success_rate":    vals.get("episode/success_rate"),
            "collision_rate":  vals.get("episode/collision_rate"),
            "timeout_rate":    vals.get("episode/timeout_rate"),
            "mean_min_lidar_m": vals.get("route/mean_min_lidar_m"),
        }
        self.entries.append(entry)
        if len(self.entries) % 10 == 0:
            self._flush()
        return True

    def _flush(self):
        try:
            dirpath = os.path.dirname(self.log_file)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(self.log_file, "w") as f:
                json.dump(self.entries, f, indent=2)
        except Exception as e:
            print(f"JSON log yazma hatasi: {e}")

    def _on_training_end(self):
        self._flush()


class PlotSaverCallback(BaseCallback):
    """
    Plots three clean figures every `save_freq` steps:
      1. Training overview  — reward, outcome rates, episode length, policy diagnostics
      2. Obstacle proximity — mean min-lidar per episode, collision/safe rates
    """

    def __init__(self, save_dir, record_freq=2048, save_freq=50_000, window=100, verbose=0):
        super().__init__(verbose)
        self.save_dir    = save_dir
        self.record_freq = record_freq
        self.save_freq   = save_freq
        self.window      = window
        self.history     = []

        self._ep_successes  = deque(maxlen=window)
        self._ep_collisions = deque(maxlen=window)
        self._ep_timeouts   = deque(maxlen=window)
        self._ep_lengths    = deque(maxlen=window)
        self._ep_rewards    = deque(maxlen=window)
        self._ep_min_lidar  = deque(maxlen=window)

    def _on_step(self) -> bool:
        if not HAS_MATPLOTLIB:
            return True

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            if i < len(dones) and dones[i]:
                self._ep_successes.append(1.0 if info.get("success") else 0.0)
                self._ep_collisions.append(1.0 if info.get("collision") else 0.0)
                self._ep_timeouts.append(1.0 if info.get("timeout") else 0.0)
                ep = info.get("episode")
                if ep:
                    self._ep_lengths.append(ep.get("l", 0))
                    self._ep_rewards.append(ep.get("r", 0.0))
                self._ep_min_lidar.append(float(info.get("min_lidar_m", 30.0)))

        if self.num_timesteps % self.record_freq == 0 and self.num_timesteps > 0:
            self._record()
        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            self._save_plots()
        return True

    def _record(self):
        vals   = getattr(self.model.logger, "name_to_value", {})
        ep_buf = list(self.model.ep_info_buffer) if hasattr(self.model, "ep_info_buffer") else []
        entry  = {
            "timesteps":       self.num_timesteps,
            "ep_rew_mean":     float(np.mean([e["r"] for e in ep_buf])) if ep_buf else 0.0,
            "ep_len_mean":     float(np.mean([e["l"] for e in ep_buf])) if ep_buf else 0.0,
            "entropy":         vals.get("train/entropy_loss") or vals.get("train/entropy"),
            "value_loss":      vals.get("train/value_loss"),
            "approx_kl":       vals.get("train/approx_kl"),
            "success_rate":    float(np.mean(self._ep_successes))  if self._ep_successes  else 0.0,
            "collision_rate":  float(np.mean(self._ep_collisions)) if self._ep_collisions else 0.0,
            "timeout_rate":    float(np.mean(self._ep_timeouts))   if self._ep_timeouts   else 0.0,
            "mean_min_lidar_m": float(np.mean(self._ep_min_lidar)) if self._ep_min_lidar  else 30.0,
        }
        self.history.append(entry)

    def _save_plots(self):
        if not self.history:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        ts_x = [h["timesteps"] for h in self.history]
        step  = self.num_timesteps

        # ── Figure 1: Training overview (2×3) ─────────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(f"Training overview — {step:,} steps", fontsize=12)

        ax = axes[0, 0]
        ax.plot(ts_x, [h["ep_rew_mean"] for h in self.history], "b-")
        ax.set_title("Mean episode reward")
        ax.set_xlabel("Timesteps")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(ts_x, [h["success_rate"]   for h in self.history], "g-",      label="Success")
        ax.plot(ts_x, [h["collision_rate"] for h in self.history], "r-",      label="Collision")
        ax.plot(ts_x, [h["timeout_rate"]   for h in self.history], color="orange", label="Timeout")
        ax.set_title("Episode outcome rates")
        ax.set_xlabel("Timesteps")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        ax.plot(ts_x, [h["ep_len_mean"] for h in self.history], color="purple")
        ax.set_title("Mean episode length")
        ax.set_xlabel("Timesteps")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ent_x = [h["timesteps"] for h in self.history if h.get("entropy") is not None]
        ent_y = [h["entropy"]   for h in self.history if h.get("entropy") is not None]
        if ent_x:
            ax.plot(ent_x, ent_y, "b-")
        ax.set_title("Policy entropy")
        ax.set_xlabel("Timesteps")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        vl_x = [h["timesteps"]  for h in self.history if h.get("value_loss") is not None]
        vl_y = [h["value_loss"] for h in self.history if h.get("value_loss") is not None]
        if vl_x:
            ax.plot(vl_x, vl_y, "r-")
        ax.set_title("Value loss")
        ax.set_xlabel("Timesteps")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        kl_x = [h["timesteps"]  for h in self.history if h.get("approx_kl") is not None]
        kl_y = [h["approx_kl"] for h in self.history if h.get("approx_kl") is not None]
        if kl_x:
            ax.plot(kl_x, kl_y, color="teal")
        ax.set_title("Approx KL divergence")
        ax.set_xlabel("Timesteps")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(self.save_dir, f"overview_{step}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── Figure 2: Obstacle proximity ──────────────────────────────────
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
        fig2.suptitle(f"Obstacle proximity — {step:,} steps", fontsize=12)

        ax = axes2[0]
        ax.plot(ts_x, [h["mean_min_lidar_m"] for h in self.history], color="darkorange")
        ax.axhline(y=2.0, color="red", linestyle="--", linewidth=0.8, label="WARN threshold (2m)")
        ax.axhline(y=0.6, color="darkred", linestyle=":", linewidth=0.8, label="Collision threshold (0.6m)")
        ax.set_title("Mean min lidar per episode (m)")
        ax.set_xlabel("Timesteps")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes2[1]
        ax.stackplot(
            ts_x,
            [h["success_rate"]   for h in self.history],
            [h["collision_rate"] for h in self.history],
            [h["timeout_rate"]   for h in self.history],
            labels=["Success", "Collision", "Timeout"],
            colors=["#4CAF50", "#F44336", "#FF9800"],
            alpha=0.75,
        )
        ax.set_title("Outcome composition")
        ax.set_xlabel("Timesteps")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out2 = os.path.join(self.save_dir, f"proximity_{step}.png")
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig2)

        if self.verbose:
            print(f"Grafikler kaydedildi: {out}, {out2}")

    def _on_training_end(self):
        if HAS_MATPLOTLIB and self.history:
            self._save_plots()


TOTAL_TIMESTEPS = int(os.environ.get("ROUTE_TOTAL_TIMESTEPS", "550000"))


def main(args=None):
    global _model, _env, _save_dir, _trajectory_recorder
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    if not rclpy.ok():
        rclpy.init(args=args)

    signal.signal(signal.SIGINT, _signal_handler)
    atexit.register(_do_shutdown)

    try:
        import torch
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    except ImportError:
        pass

    run_name  = f"RouteCurriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    models_dir = os.path.abspath(os.path.join("models", run_name))
    log_dir    = os.path.abspath(os.path.join("logs",   run_name))
    traj_dir   = os.path.abspath(os.path.join(log_dir,  "trajectories"))
    plots_dir  = os.path.abspath(os.path.join(log_dir,  "plots"))
    for d in (models_dir, log_dir, traj_dir, plots_dir):
        os.makedirs(d, exist_ok=True)
    _save_dir = models_dir

    old = glob.glob(f"{models_dir}/*.zip")
    if old:
        raise RuntimeError(
            f"Eski checkpoint var, sıfırdan başlamak için önce sil:\n"
            f"  rm -rf {models_dir}\n"
            f"Bulunanlar: {old}"
        )

    print("=" * 60, flush=True)
    print("ROUTE AGENT CURRICULUM TRAINING", flush=True)
    print("=" * 60, flush=True)
    print(f"Run:     {run_name}", flush=True)
    print(f"Models:  {models_dir}", flush=True)
    print(f"Logs:    {log_dir}", flush=True)
    print(f"Total:   {TOTAL_TIMESTEPS:,} timesteps", flush=True)
    if UNIFIED_MAZE:
        print("Mode:    BIRLESIK MAZE (pozisyon bazli stage)", flush=True)
    else:
        sr = CurriculumScheduler.default_stage_ranges()
        print(f"Stages:  1 {sr[1]} | 2 {sr[2]} | 3 {sr[3]}", flush=True)
    print("Reward:  +2 progress | +1 safe | +100 goal | -5 too-close | -0.1/step | -100 crash",
          flush=True)
    if not HAS_MATPLOTLIB:
        print("Not:     Grafik icin: pip install matplotlib", flush=True)
    print("=" * 60, flush=True)

    scheduler_node = None
    stage_pub      = None
    ros_thread     = None
    if not UNIFIED_MAZE:
        scheduler_node = rclpy.create_node("train_route_curriculum_scheduler")
        stage_pub      = scheduler_node.create_publisher(Int32, "/route/set_stage", 10)
        ros_thread     = threading.Thread(target=rclpy.spin, args=(scheduler_node,), daemon=True)
        ros_thread.start()

    def make_env():
        env = gym.make("RouteCurriculumAgent-v0")
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        env = TimestepSyncWrapper(env)
        return env

    vec_env = DummyVecEnv([make_env])

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

    env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=HYPERPARAMS["gamma"],
    )
    _env = env

    from uav_route_planner.networks.route_extractor import RouteCombinedExtractor

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        **HYPERPARAMS,
        policy_kwargs=dict(
            features_extractor_class=RouteCombinedExtractor,
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        ),
    )
    _model = model

    callbacks = CallbackList([
        SyncSB3TimestepCallback(),
        CurriculumScheduler(verbose=1, stage_pub=stage_pub),
        RouteTrainingMonitor(log_freq=2048, window=100),
        TrajectoryRecorder(save_dir=traj_dir, max_episodes=200),
        ProgressReporter(
            total_timesteps=TOTAL_TIMESTEPS,
            report_freq=int(os.environ.get("TRAIN_PROGRESS_FREQ", "1000")),
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=models_dir,
            name_prefix="route_curriculum",
            save_vecnormalize=True,
            verbose=1,
        ),
        TrainingLogWriter(
            log_file=os.path.join(log_dir, "training_log.json"),
            log_freq=5_000,
        ),
        PlotSaverCallback(
            save_dir=plots_dir,
            record_freq=2048,
            save_freq=50_000,
            verbose=1,
        ),
    ])

    print("\nEgitim basliyor...\n", flush=True)

    SAVE_INTERVAL = 50_000
    steps_done = 0
    _trajectory_recorder = callbacks.callbacks[3]  # TrajectoryRecorder

    while steps_done < TOTAL_TIMESTEPS:
        chunk = min(SAVE_INTERVAL, TOTAL_TIMESTEPS - steps_done)
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            tb_log_name="RouteCurriculum",
            callback=callbacks,
        )
        steps_done += chunk

    model.save(os.path.join(models_dir, "final_model"))
    env.save(os.path.join(models_dir, "vec_normalize_final.pkl"))
    _trajectory_recorder.flush()
    callbacks.callbacks[7]._on_training_end()  # PlotSaverCallback

    print("\n" + "=" * 60)
    print("EGITIM TAMAMLANDI")
    print(f"Model:  {models_dir}/final_model.zip")
    print(f"Grafik: {plots_dir}/")
    print("=" * 60)

    atexit.unregister(_do_shutdown)
    env.close()
    if scheduler_node is not None:
        try:
            scheduler_node.destroy_node()
        except Exception:
            pass
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()