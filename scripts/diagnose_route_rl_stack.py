#!/usr/bin/env python3
"""
Route RL stack — tartışılan sorunları ölç: waypoint gecikmesi, filtre farkı, topic zinciri.

Ne ölçer?
  1) /route/waypoint_desired ardışık mesaj aralığı (follow_path ROUTE_WP_TIMEOUT=0.8 ile kıyas)
  2) /route/waypoint_desired vs /route/waypoint_safe konum farkı (safety filter aktifse)
  3) /threat/state_vec ve /odometry/filtered mesaj sıklığı (kaba)
  4) offboard_control düğümü (follow_path executable) varsa route_waypoint_topic parametresi

Kullanım (sim + ilgili node'lar çalışırken):
  source install/setup.bash   # veya workspace setup.bash
  python3 scripts/diagnose_route_rl_stack.py --duration 30

  python3 scripts/diagnose_route_rl_stack.py -d 60 --timeout-threshold 0.8 --gap-alarm-ratio 0.05

Çıktı: UYARI satırları — eşik aşımı veya topic yok.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray


def _hypot_xy(x: float, y: float) -> float:
    return math.hypot(x, y)


@dataclass
class StreamStats:
    name: str
    inter_arrival: Deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    last_recv: float = 0.0
    last_x: float = 0.0
    last_y: float = 0.0
    last_z: float = 0.0
    count: int = 0
    gaps_over_threshold: int = 0
    thr_ref: float = 0.8

    def on_msg(self, now: float, x: float, y: float, z: float, gap_threshold: float) -> None:
        if self.count > 0 and self.last_recv > 0:
            dt = now - self.last_recv
            self.inter_arrival.append(dt)
            if dt > gap_threshold:
                self.gaps_over_threshold += 1
        self.last_recv = now
        self.last_x = x
        self.last_y = y
        self.last_z = z
        self.count += 1

    def summary(self) -> str:
        if not self.inter_arrival:
            return f"{self.name}: tek mesaj veya yok (ardışık aralık yok)"
        arr = list(self.inter_arrival)
        return (
            f"{self.name}: n={self.count}, aralık s min/mean/max = "
            f"{min(arr):.3f} / {sum(arr)/len(arr):.3f} / {max(arr):.3f} s, "
            f">{self.thr_ref:.2f}s sayısı={self.gaps_over_threshold}"
        )


class DiagnoseNode(Node):
    def __init__(
        self,
        duration_sec: float,
        gap_threshold: float,
        follow_path_name: str,
    ):
        super().__init__("diagnose_route_rl_stack")
        self.duration_sec = duration_sec
        self.gap_threshold = gap_threshold
        self.follow_path_name = follow_path_name
        self._start = time.monotonic()

        qos_best = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.desired = StreamStats("/route/waypoint_desired", thr_ref=gap_threshold)
        self.safe = StreamStats("/route/waypoint_safe", thr_ref=gap_threshold)
        self.threat = StreamStats("/threat/state_vec", thr_ref=gap_threshold)
        self.odom = StreamStats("/odometry/filtered", thr_ref=gap_threshold)

        self.delta_xy: Deque[float] = deque(maxlen=2000)
        self.pairs_compared = 0

        self.create_subscription(
            PoseStamped,
            "/route/waypoint_desired",
            self._cb_desired,
            10,
        )
        self.create_subscription(
            PoseStamped,
            "/route/waypoint_safe",
            self._cb_safe,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            "/threat/state_vec",
            self._cb_threat,
            qos_best,
        )
        self.create_subscription(
            Odometry,
            "/odometry/filtered",
            self._cb_odom,
            qos_best,
        )

        self.create_timer(0.5, self._tick_status)

    def _now(self) -> float:
        return time.monotonic()

    def _cb_desired(self, msg: PoseStamped) -> None:
        now = self._now()
        p = msg.pose.position
        self.desired.on_msg(now, p.x, p.y, p.z, self.gap_threshold)

    def _cb_safe(self, msg: PoseStamped) -> None:
        now = self._now()
        p = msg.pose.position
        self.safe.on_msg(now, p.x, p.y, p.z, self.gap_threshold)

        if self.desired.count > 0:
            d = _hypot_xy(p.x - self.desired.last_x, p.y - self.desired.last_y)
            self.delta_xy.append(d)
            self.pairs_compared += 1

    def _cb_threat(self, msg: Float32MultiArray) -> None:
        now = self._now()
        self.threat.on_msg(now, 0.0, 0.0, 0.0, self.gap_threshold)

    def _cb_odom(self, msg: Odometry) -> None:
        now = self._now()
        p = msg.pose.pose.position
        self.odom.on_msg(now, p.x, p.y, p.z, self.gap_threshold)

    def _tick_status(self) -> None:
        elapsed = self._now() - self._start
        if elapsed >= self.duration_sec:
            self._final_report()
            rclpy.shutdown()

    def _final_report(self) -> None:
        print("\n" + "=" * 72)
        print("ROUTE RL STACK — TANI ÖZETİ")
        print("=" * 72)
        print(
            f"Süre: {self.duration_sec:.1f} s | "
            f"Waypoint aralık alarm eşiği: {self.gap_threshold:.2f} s "
            f"(follow_path ROUTE_WP_TIMEOUT ile karşılaştır; varsayılan kodda 0.8)"
        )
        print()

        for s in (self.desired, self.safe, self.threat, self.odom):
            print(s.summary())

        print()
        if self.desired.count == 0:
            print(
                "UYARI: /route/waypoint_desired mesajı yok — RL/train node çalışmıyor veya "
                "topic adı farklı."
            )
        elif self.desired.gaps_over_threshold > 0:
            print(
                f"UYARI: desired üzerinde {self.desired.gaps_over_threshold} kez "
                f"ardışık mesaj aralığı > {self.gap_threshold:.2f} s — follow_path "
                f"bu sürede waypoint saymayı bırakabilir (donma riski)."
            )
        else:
            print(
                "desired: ardışık aralıklar eşik altında görünüyor (timeout riski düşük, "
                "ölçüm süresine bağlı)."
            )

        print()
        if self.safe.count == 0:
            print(
                "BİLGİ: /route/waypoint_safe yok — route_safety_filter_node kapalı olabilir "
                "veya hiç desired gelmediği için filtre yayınlamıyor."
            )
        elif self.delta_xy:
            dm = list(self.delta_xy)
            print(
                f"desired↔safe XY mesafe (yaklaşık, safe callback anındaki son desired ile): "
                f"min/mean/max = {min(dm):.3f} / {sum(dm)/len(dm):.3f} / {max(dm):.3f} m "
                f"(karşılaştırma sayısı {self.pairs_compared})"
            )
            if max(dm) > 0.05:
                print(
                    "NOT: Filtre desired’ı kaydırıyorsa birkaç cm–m arası fark normal; "
                    "hep büyükse maliyet/tehdit şişirmesi veya sık düzeltme olabilir."
                )

        print()
        _print_follow_path_param(self.follow_path_name)
        _print_ros2_topic_brief()

        print("=" * 72)


def _print_follow_path_param(node_name: str) -> None:
    try:
        out = subprocess.run(
            [
                "ros2",
                "param",
                "get",
                f"/{node_name.lstrip('/')}",
                "route_waypoint_topic",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0:
            print(f"follow_path parametresi route_waypoint_topic: {out.stdout.strip()}")
        else:
            print(
                f"BİLGİ: follow_path düğümü bulunamadı veya param yok: {node_name}\n"
                f"  ({out.stderr.strip() or out.stdout.strip()})"
            )
    except FileNotFoundError:
        print("BİLGİ: ros2 CLI yok — parametre kontrolü atlandı.")
    except subprocess.TimeoutExpired:
        print("BİLGİ: ros2 param get zaman aşımı.")


def _print_ros2_topic_brief() -> None:
    try:
        out = subprocess.run(
            ["ros2", "topic", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return
        topics = out.stdout.splitlines()
        need = [
            "/route/waypoint_desired",
            "/route/waypoint_safe",
            "/threat/state_vec",
            "/odometry/filtered",
            "/local_costmap/costmap",
        ]
        print("Topic var mı (hızlı kontrol):")
        for t in need:
            ok = t in topics
            print(f"  {'+' if ok else '-'} {t}")
    except FileNotFoundError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Route RL stack waypoint/filtre/timeout tanısı"
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=30.0,
        help="Örnekleme süresi (saniye)",
    )
    parser.add_argument(
        "--timeout-threshold",
        type=float,
        default=0.8,
        help="follow_path ROUTE_WP_TIMEOUT ile aynı mantıkta alarm (s)",
    )
    parser.add_argument(
        "--follow-path-node",
        type=str,
        default="offboard_control",
        help=(
            "ros2 param get için node adı (follow_path.py içinde super().__init__('offboard_control'))"
        ),
    )
    args = parser.parse_args()

    rclpy.init()
    node = DiagnoseNode(
        duration_sec=args.duration,
        gap_threshold=args.timeout_threshold,
        follow_path_name=args.follow_path_node,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nKesildi.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
