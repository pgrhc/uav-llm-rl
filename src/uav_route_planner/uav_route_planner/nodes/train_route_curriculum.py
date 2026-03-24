#!/usr/bin/env python3
"""
train_route_curriculum.py — 3-stage curriculum training for Route Agent

Stages (non-unified; varsayılan toplam 300K — env ile değişir):
    1  (0 - 75K)    Path following — open maze, no actors
    2  (75K - 175K) Static obstacles — narrow corridors, dead-ends
    3  (175K - 300K) Dynamic threats — 180-200 walking actors

Callbacks:
    CurriculumScheduler     Stage transition + lazy actor spawning
    RouteTrainingMonitor    Policy entropy, action std, value loss,
                            success/collision/timeout rates, episode length,
                            deterministic-vs-stochastic comparison,
                            route quality metrics (path error, threat exposure),
                            reward/mean_rw_* (episode means of env info components)
    TrajectoryRecorder      Per-episode position + entropy/std for heatmap
    ProgressReporter        ETA and FPS

Usage:
    ros2 run uav_route_planner train_route_curriculum

VecEnv:
    Varsayılan SubprocVecEnv(spawn): ROS env ayrı süreçte (GIL ile SAC ayrılır).
    Sorun çıkarsa: ROUTE_USE_DUMMY_VEC=1 ile eski tek-süreç DummyVecEnv.

SAC başlangıç (kök neden — “drone ne yapacağını bilmiyor” hissi):
    learning_starts (ROUTE_LEARNING_STARTS, varsayılan 5000) dolana kadar SB3 SAC rastgele
    aksiyon toplar; politika henüz anlamlı değildir. Bu fazda waypoint’ler sarsılır.
    İnce ayar: ROUTE_RANDOM_PHASE_ACTION_SCALE=0.25 — sadece bu fazda aksiyon ölçeği (env içinde).
    train script SyncSB3TimestepCallback ile _sb3_num_timesteps yazar (Subproc uyumlu).

Waypoint adımı:
    ROUTE_STEP_SIZE (m, varsayılan 0.3) — RouteCurriculumEnv / RouteEnv xy residual ölçeği; 0.2 tipik yumuşatma.

Torch iş parçacığı:
    TORCH_NUM_THREADS (varsayılan 1) — ana süreçte CPU çekişmesini azaltmak için.
"""

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
    matplotlib.use("Agg")  # GUI yok, sadece dosyaya kaydet
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    # Grafik kaydetmek icin: pip install matplotlib
import gymnasium as gym
import rclpy

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

import uav_route_planner.envs  # noqa: F401  — triggers register()

# Birlesik maze: pozisyon bazli stage, teleport yok
UNIFIED_MAZE = True

# Stage 3 aktor spawn: birlesik maze'de maze ile birlikte spawn edilir
def spawn_stage3_actors_lazy():
    try:
        from uav_route_planner.maze_curriculum_world import spawn_stage3_actors_lazy as _fn
        return _fn()
    except ModuleNotFoundError:
        print(
            "WARNING: maze_curriculum_world yok. Stage 3 aktorleri spawn edilmedi. "
            "Aktorler icin: ros2 run uav_route_planner maze_curriculum_world --actors"
        )
        return []

from std_msgs.msg import Int32


class TimestepSyncWrapper(gym.Wrapper):
    """Monitor dışında: VecEnv.env_method('set_sb3_timesteps', n) → RouteCurriculumEnv."""

    def set_sb3_timesteps(self, n: int) -> None:
        self.env.unwrapped.set_sb3_timesteps(int(n))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class SyncSB3TimestepCallback(BaseCallback):
    """Her adımda SB3 toplam timestep’i env’e iletir (random-phase ölçekleme için)."""

    def _on_step(self) -> bool:
        try:
            self.training_env.env_method("set_sb3_timesteps", int(self.num_timesteps))
        except Exception:
            pass
        return True


def _route_curriculum_subproc_env_fn(log_dir: str, rank: int):
    """
    SubprocVecEnv (spawn) worker: ROS ayrı süreçte → GIL, SAC/PyTorch'tan ayrı.
    Worker sürecinde Gym kaydı için mutlaka env paketi import edilmeli.
    """
    def _init():
        import uav_route_planner.envs  # noqa: F401 — RouteCurriculumAgent-v0 register (spawn'ta şart)

        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"monitor_{rank}.csv")
        env = gym.make("RouteCurriculumAgent-v0")
        env = Monitor(env, filename=path)
        env = TimestepSyncWrapper(env)
        return env

    return _init


def _vec_env_call_method(training_env, method_name: str, *args, indices=None):
    """VecNormalize altındaki DummyVecEnv / SubprocVecEnv üzerinde metod çağrısı."""
    venv = training_env.venv
    idx = indices if indices is not None else [0]
    return venv.env_method(method_name, *args, indices=idx)[0]


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBALS FOR SIGNAL HANDLER (protected by lock to avoid race with training loop)
# ═══════════════════════════════════════════════════════════════════════════════

_shutdown_lock = threading.Lock()
_shutdown_done = False
_model = None
_env = None
_save_dir = None
_trajectory_recorder = None


def _do_shutdown():
    """Save and cleanup. Called by signal handler or atexit."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK 1: CURRICULUM SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

class CurriculumScheduler(BaseCallback):
    """Manages stage transitions and lazy actor spawning."""

    @staticmethod
    def default_stage_ranges():
        """ROUTE_TOTAL_TIMESTEPS, ROUTE_STAGE1_END, ROUTE_STAGE2_END ile yapılandırılır."""
        t = int(os.environ.get("ROUTE_TOTAL_TIMESTEPS", "300000"))
        s1 = int(os.environ.get("ROUTE_STAGE1_END", "75000"))
        s2 = int(os.environ.get("ROUTE_STAGE2_END", "175000"))
        s1 = max(0, min(s1, t))
        s2 = max(s1, min(s2, t))
        return {1: (0, s1), 2: (s1, s2), 3: (s2, t)}

    def __init__(self, verbose=1, stage_pub=None):
        super().__init__(verbose)
        self.current_stage = 1
        self.stage_ranges = CurriculumScheduler.default_stage_ranges()
        # Non-unified: ana süreçte /route/set_stage (Subproc'te child node'a erişilmez)
        self._stage_pub = stage_pub
        self._actors_spawned = False

    def _on_training_start(self):
        if UNIFIED_MAZE:
            return  # Stage from position; no publisher needed
        if self._stage_pub is not None:
            return
        raw_env = self._get_raw_env()
        if raw_env and hasattr(raw_env, "node"):
            self._stage_pub = raw_env.node.create_publisher(
                Int32, "/route/set_stage", 10
            )

    def _on_step(self) -> bool:
        if UNIFIED_MAZE:
            return True  # Birlesik maze: stage pozisyondan, scheduler devre disi
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


# Per-step keys from route_curriculum_env info (episode means → TensorBoard reward/mean_*).
_ROUTE_RW_INFO_KEYS = (
    "rw_progress",
    "rw_goal",
    "rw_collision",
    "rw_path",
    "rw_astar_return",
    "rw_path_drift",
    "rw_threat",
    "rw_smooth",
    "rw_time",
)


# ═══════════════════════════════════════════════════════════════════════════════
# BASE: Route Episode Metrics (shared by Monitor + PlotSaver)
# ═══════════════════════════════════════════════════════════════════════════════

class _RouteEpisodeMetricsMixin:
    """Shared episode metrics collection. Used by RouteTrainingMonitor and PlotSaverCallback."""

    def _init_episode_metrics(self, window=100):
        self.ep_successes = deque(maxlen=window)
        self.ep_collisions = deque(maxlen=window)
        self.ep_timeouts = deque(maxlen=window)
        self.ep_lengths = deque(maxlen=window)
        self.ep_rewards = deque(maxlen=window)
        self.ep_path_errors = deque(maxlen=window)
        self.ep_threat_maxes = deque(maxlen=window)
        self.ep_threat_gate_means = deque(maxlen=window)
        self.ep_astar_follow_rates = deque(maxlen=window)
        self.ep_rw_means = {k: deque(maxlen=window) for k in _ROUTE_RW_INFO_KEYS}
        self._step_path_errors = []
        self._step_threat_maxes = []
        self._step_threat_gates = []
        self._step_on_astar = []
        self._step_rw = {k: [] for k in _ROUTE_RW_INFO_KEYS}

    def _collect_episode_metrics(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            self._step_path_errors.append(info.get("path_error", 0.0))
            self._step_threat_maxes.append(info.get("max_threat", 0.0))
            self._step_threat_gates.append(float(info.get("threat_gate", 0.0)))
            self._step_on_astar.append(float(info.get("on_astar", 0.0)))
            for k in _ROUTE_RW_INFO_KEYS:
                self._step_rw[k].append(float(info.get(k, 0.0)))
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
                if self._step_threat_gates:
                    self.ep_threat_gate_means.append(np.mean(self._step_threat_gates))
                if self._step_on_astar:
                    self.ep_astar_follow_rates.append(np.mean(self._step_on_astar))
                for k in _ROUTE_RW_INFO_KEYS:
                    buf = self._step_rw[k]
                    if buf:
                        self.ep_rw_means[k].append(float(np.mean(buf)))
                for k in _ROUTE_RW_INFO_KEYS:
                    self._step_rw[k].clear()
                self._step_path_errors.clear()
                self._step_threat_maxes.clear()
                self._step_threat_gates.clear()
                self._step_on_astar.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK 2: ROUTE TRAINING MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class RouteTrainingMonitor(_RouteEpisodeMetricsMixin, BaseCallback):
    """
    Unified monitoring callback:
    - Policy entropy, action std, value loss
    - Success / collision / timeout rates
    - Episode length, mean reward
    - Deterministic vs stochastic action comparison
    - Route quality: path error, threat exposure
    """

    def __init__(self, log_freq=2048, compare_freq=10_000, window=100, verbose=0):
        BaseCallback.__init__(self, verbose)
        self.log_freq = log_freq
        self.compare_freq = compare_freq
        self.window = window
        self._init_episode_metrics(window)

    def _on_step(self) -> bool:
        self._collect_episode_metrics()

        if self.num_timesteps % self.log_freq == 0:
            self._log_training_metrics()
            self._log_episode_metrics()

        if self.num_timesteps % self.compare_freq == 0:
            self._log_det_vs_stoch()

        return True

    def _log_training_metrics(self):
        vals = getattr(self.model.logger, "name_to_value", {})

        # PPO: train/entropy_loss; SAC: train/entropy or train/ent_coef (ent_coef_loss is alpha loss)
        entropy = vals.get("train/entropy_loss") or vals.get("train/entropy") or vals.get("train/ent_coef")
        if entropy is not None:
            self.logger.record("monitor/policy_entropy", entropy)

        # PPO: train/value_loss; SAC: train/critic_loss
        value_loss = vals.get("train/value_loss") or vals.get("train/critic_loss")
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
        if len(self.ep_threat_gate_means) > 0:
            self.logger.record(
                "route/mean_threat_gate", np.mean(self.ep_threat_gate_means)
            )
        if len(self.ep_astar_follow_rates) > 0:
            self.logger.record(
                "route/astar_follow_rate", np.mean(self.ep_astar_follow_rates)
            )
        for k in _ROUTE_RW_INFO_KEYS:
            dq = self.ep_rw_means[k]
            if len(dq) > 0:
                self.logger.record(f"reward/mean_{k}", float(np.mean(dq)))

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
        try:
            pos = _vec_env_call_method(self.training_env, "get_drone_position")
        except Exception:
            return True
        if pos is None:
            return True
        arr = np.asarray(pos, dtype=np.float64).ravel()
        if arr.size < 3:
            return True
        x, y, z = float(arr[0]), float(arr[1]), float(arr[2])
        self._current_positions.append([x, y, z])

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
        self.last_reported_step = 0

    def _on_training_start(self):
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.last_reported_step = 0
        print(
            f"[train] learn dongusu basladi. Rapor her {self.report_freq:,} adimda (FPS/ETA).",
            flush=True,
        )

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.last_reported_step + self.report_freq:
            self._report()
            self.last_reported_step = self.num_timesteps
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
            f"| FPS: {fps:.0f} | ETA: {h}h {m}m",
            flush=True,
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
            "entropy": vals.get("train/entropy_loss") or vals.get("train/entropy") or vals.get("train/ent_coef"),
            "value_loss": vals.get("train/value_loss") or vals.get("train/critic_loss"),
            "approx_kl": vals.get("train/approx_kl", None),
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


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK 6: PLOT SAVER — Grafikleri PNG olarak kaydet
# ═══════════════════════════════════════════════════════════════════════════════

class PlotSaverCallback(_RouteEpisodeMetricsMixin, BaseCallback):
    """Metrikleri grafik olarak PNG dosyasina kaydeder."""

    def __init__(self, save_dir, record_freq=2048, save_freq=50_000, window=100, verbose=0):
        BaseCallback.__init__(self, verbose)
        self.save_dir = save_dir
        self.record_freq = record_freq
        self.save_freq = save_freq
        self.window = window
        self._init_episode_metrics(window)
        self.history = []  # [(timestep, {...}), ...]

    def _on_step(self) -> bool:
        if not HAS_MATPLOTLIB:
            return True

        self._collect_episode_metrics()

        if self.num_timesteps % self.record_freq == 0 and self.num_timesteps > 0:
            self._record()
        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            self._save_plots()

        return True

    def _record(self):
        vals = getattr(self.model.logger, "name_to_value", {})
        ep_buf = list(self.model.ep_info_buffer) if hasattr(self.model, "ep_info_buffer") else []

        entry = {
            "timesteps": self.num_timesteps,
            "entropy": vals.get("train/entropy_loss") or vals.get("train/entropy") or vals.get("train/ent_coef"),
            "value_loss": vals.get("train/value_loss") or vals.get("train/critic_loss"),
            "approx_kl": vals.get("train/approx_kl"),
            "ep_rew_mean": float(np.mean([e["r"] for e in ep_buf])) if ep_buf else 0.0,
            "ep_len_mean": float(np.mean([e["l"] for e in ep_buf])) if ep_buf else 0.0,
            "success_rate": float(np.mean(self.ep_successes)) if self.ep_successes else 0.0,
            "collision_rate": float(np.mean(self.ep_collisions)) if self.ep_collisions else 0.0,
            "timeout_rate": float(np.mean(self.ep_timeouts)) if self.ep_timeouts else 0.0,
            "path_error": float(np.mean(self.ep_path_errors)) if self.ep_path_errors else 0.0,
            "threat_exposure": float(np.mean(self.ep_threat_maxes)) if self.ep_threat_maxes else 0.0,
            "mean_threat_gate": float(np.mean(self.ep_threat_gate_means)) if self.ep_threat_gate_means else 0.0,
            "astar_follow_rate": float(np.mean(self.ep_astar_follow_rates)) if self.ep_astar_follow_rates else 0.0,
        }
        for k in _ROUTE_RW_INFO_KEYS:
            dq = self.ep_rw_means[k]
            entry[f"mean_{k}"] = float(np.mean(dq)) if dq else 0.0
        self.history.append(entry)

    def _save_plots(self):
        if not self.history:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        ts = [h["timesteps"] for h in self.history]

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle(f"Route Curriculum Training — {self.num_timesteps:,} steps", fontsize=12)

        # Episode metrics
        ax = axes[0, 0]
        ax.plot(ts, [h["ep_rew_mean"] for h in self.history], "b-", label="Reward")
        ax.set_title("Mean Episode Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(ts, [h["success_rate"] for h in self.history], "g-", label="Success")
        ax.plot(ts, [h["collision_rate"] for h in self.history], "r-", label="Collision")
        ax.plot(ts, [h["timeout_rate"] for h in self.history], "orange", label="Timeout")
        ax.set_title("Outcome Rates")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        ax.plot(ts, [h["ep_len_mean"] for h in self.history], "purple")
        ax.set_title("Mean Episode Length")
        ax.grid(True, alpha=0.3)

        # Training metrics
        ax = axes[1, 0]
        ent_ts = [h["timesteps"] for h in self.history if h["entropy"] is not None]
        ent_vals = [h["entropy"] for h in self.history if h["entropy"] is not None]
        if ent_ts and ent_vals:
            ax.plot(ent_ts, ent_vals, "b-")
        ax.set_title("Policy Entropy")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        vl_ts = [h["timesteps"] for h in self.history if h["value_loss"] is not None]
        vl_vals = [h["value_loss"] for h in self.history if h["value_loss"] is not None]
        if vl_ts and vl_vals:
            ax.plot(vl_ts, vl_vals, "r-")
        ax.set_title("Value Loss")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        ax.plot(ts, [h["path_error"] for h in self.history], "b-", label="Path Error")
        ax.plot(ts, [h["threat_exposure"] for h in self.history], "r-", label="Threat")
        ax.plot(ts, [h.get("astar_follow_rate", 0.0) for h in self.history], "g--", label="A* follow")
        ax.set_title("Route Quality")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.save_dir, f"training_plots_{self.num_timesteps}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        if self.verbose:
            print(f"Grafikler kaydedildi: {path}")

        # Episode ortalaması ödül bileşenleri (env info rw_* → rolling window)
        fig_rw, axes_rw = plt.subplots(3, 3, figsize=(12, 10))
        fig_rw.suptitle(
            f"Reward components (mean per episode, window) — {self.num_timesteps:,} steps",
            fontsize=11,
        )
        for ax, k in zip(axes_rw.flat, _ROUTE_RW_INFO_KEYS):
            key = f"mean_{k}"
            ax.plot(ts, [h.get(key, 0.0) for h in self.history], linewidth=1.0)
            ax.set_title(k, fontsize=9)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path_rw = os.path.join(
            self.save_dir, f"training_reward_rw_{self.num_timesteps}.png"
        )
        plt.savefig(path_rw, dpi=150, bbox_inches="tight")
        plt.close(fig_rw)
        if self.verbose:
            print(f"Ödül bileşen grafikleri: {path_rw}")

    def _on_training_end(self):
        if HAS_MATPLOTLIB and self.history:
            self._save_plots()
            if self.verbose:
                print(f"Final grafikler: {self.save_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

TOTAL_TIMESTEPS = int(os.environ.get("ROUTE_TOTAL_TIMESTEPS", "300000"))


def main(args=None):
    global _model, _env, _save_dir, _trajectory_recorder

    # noVNC / pipe / bazı terminallerde stdout tamponlu kalır → "log yok" sanılır
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

    run_name = f"RouteCurriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    models_dir = os.path.abspath(os.path.join("models", run_name))
    log_dir = os.path.abspath(os.path.join("logs", run_name))
    traj_dir = os.path.abspath(os.path.join(log_dir, "trajectories"))
    plots_dir = os.path.abspath(os.path.join(log_dir, "plots"))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    _save_dir = models_dir

    print("=" * 60, flush=True)
    print("ROUTE AGENT CURRICULUM TRAINING", flush=True)
    print("=" * 60, flush=True)
    print(f"Run:     {run_name}", flush=True)
    print(f"Models:  {models_dir}", flush=True)
    print(f"Logs:    {log_dir}", flush=True)
    print(f"Total:   {TOTAL_TIMESTEPS:,} timesteps", flush=True)
    if UNIFIED_MAZE:
        print(f"Mode:    BIRLESIK MAZE (pozisyon bazli stage, teleport yok)", flush=True)
    else:
        sr = CurriculumScheduler.default_stage_ranges()
        print(
            f"Stages:  1 {sr[1]} | 2 {sr[2]} | 3 {sr[3]} (env: ROUTE_STAGE*_END)",
            flush=True,
        )
    if not HAS_MATPLOTLIB:
        print("Not:    Grafik kaydi icin: pip install matplotlib", flush=True)
    use_dummy_vec = os.environ.get("ROUTE_USE_DUMMY_VEC", "").lower() in ("1", "true", "yes")
    if use_dummy_vec:
        print(
            "VecEnv:  DummyVecEnv (ROUTE_USE_DUMMY_VEC) — ROS+SAC ayni proses; GIL riski",
            flush=True,
        )
    else:
        print(
            "VecEnv:  SubprocVecEnv(spawn) — env ayri proses; GIL'den kurtulma",
            flush=True,
        )
    print("=" * 60, flush=True)

    scheduler_node = None
    stage_pub = None
    if not UNIFIED_MAZE:
        scheduler_node = rclpy.create_node("train_route_curriculum_scheduler")
        stage_pub = scheduler_node.create_publisher(Int32, "/route/set_stage", 10)

    # --- Environment ---
    def make_env():
        env = gym.make("RouteCurriculumAgent-v0")
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        env = TimestepSyncWrapper(env)
        return env

    if use_dummy_vec:
        vec_env = DummyVecEnv([make_env])
    else:
        vec_env = SubprocVecEnv(
            [_route_curriculum_subproc_env_fn(log_dir, 0)],
            start_method="spawn",
        )
    env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    _env = env

    # --- Model (SAC + Asymmetric Policy) ---
    from uav_route_planner.networks.route_asymmetric_policy import RouteAsymmetricSACPolicy

    _ls = int(os.environ.get("ROUTE_LEARNING_STARTS", "5000"))
    _rscale = float(os.environ.get("ROUTE_RANDOM_PHASE_ACTION_SCALE", "1.0"))

    model = SAC(
        RouteAsymmetricSACPolicy,
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=_ls,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(
            pi_arch=[256, 256, 128],
            vf_arch=[256, 256, 128],
        ),
    )
    _model = model

    # --- Callbacks ---
    curriculum_scheduler = CurriculumScheduler(verbose=1, stage_pub=stage_pub)

    route_monitor = RouteTrainingMonitor(
        log_freq=2048, compare_freq=10_000, window=100,
    )

    trajectory_recorder = TrajectoryRecorder(
        save_dir=traj_dir, max_episodes=200,
    )
    _trajectory_recorder = trajectory_recorder

    progress_reporter = ProgressReporter(
        total_timesteps=TOTAL_TIMESTEPS,
        report_freq=int(os.environ.get("TRAIN_PROGRESS_FREQ", "1000")),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=models_dir,
        name_prefix="route_curriculum",
        save_vecnormalize=True,
        verbose=1,
    )

    training_log = TrainingLogWriter(
        log_file=os.path.join(log_dir, "training_log.json"),
        log_freq=5_000,
    )

    plot_saver = PlotSaverCallback(
        save_dir=plots_dir,
        record_freq=2048,
        save_freq=50_000,
        verbose=1,
    )

    timestep_sync_cb = SyncSB3TimestepCallback()

    callbacks = CallbackList([
        timestep_sync_cb,
        curriculum_scheduler,
        route_monitor,
        trajectory_recorder,
        progress_reporter,
        checkpoint_cb,
        training_log,
        plot_saver,
    ])

    # --- Train ---
    print("", flush=True)
    print("!" * 60, flush=True)
    print("SAC BASLANGIC NOTU (kok neden analizi)", flush=True)
    print(
        f"  learning_starts={_ls}  →  bu adim sayisina KADAR politika EGITILMEZ; "
        "SB3 rastgele aksiyon toplar.",
        flush=True,
    )
    print(
        "  Bu yuzden ilk binlerce adimda waypoint'ler 'anlamsiz' gorunebilir; "
        "drone da PX4/EKF gecikmesiyle sarsilir.",
        flush=True,
    )
    if _rscale < 1.0 - 1e-9:
        print(
            f"  ROUTE_RANDOM_PHASE_ACTION_SCALE={_rscale}  →  random fazda "
            "env aksiyonu bu faktorle kucultulur (daha yumusak waypoint).",
            flush=True,
        )
    else:
        print(
            "  Random fazda yumusatma icin: export ROUTE_RANDOM_PHASE_ACTION_SCALE=0.25",
            flush=True,
        )
    _step_m = float(os.environ.get("ROUTE_STEP_SIZE", "0.3"))
    print(
        f"  ROUTE_STEP_SIZE={_step_m} m  (xy residual; yumusak: 0.2, varsayilan 0.3)",
        flush=True,
    )
    print("!" * 60, flush=True)
    print(
        "\nEgitim basliyor... (ROUTE_LEARNING_STARTS / ROUTE_RANDOM_PHASE_ACTION_SCALE / ROUTE_STEP_SIZE)\n",
        flush=True,
    )

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
        # CheckpointCallback handles saves at save_freq; no manual duplicate

    # --- Final save ---
    model.save(os.path.join(models_dir, "final_model"))
    env.save(os.path.join(models_dir, "vec_normalize_final.pkl"))
    trajectory_recorder.flush()
    plot_saver._on_training_end()

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
