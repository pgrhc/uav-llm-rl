#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import PointStamped

from vision_msgs.msg import Detection3DArray

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

import numpy as np
from collections import deque

from fusion_msgs.msg import RadarPoints
from nav_msgs.msg import Odometry
import math


# ══════════════════════════════════════════════════════════════════════════════
# YOLO → Environment Class Mapping
# ══════════════════════════════════════════════════════════════════════════════
# COCO dataset class IDs → Environment expected class IDs
# COCO:        0=person, 2=car, 16=bird, ...
# Environment: 0=Unknown, 1=Drone, 2=Bird, 3=FixedWing, 4=Person
# ══════════════════════════════════════════════════════════════════════════════

YOLO_TO_ENV_CLASS = {
    "0":  4,   # COCO person   → Environment Person
    "16": 2,   # COCO bird     → Environment Bird
    "2":  1,   # COCO car      → Environment Drone (yaklaşım)
    "5":  1,   # COCO airplane → Environment Drone (yaklaşım)
    # Diğer sınıflar Unknown (0) olarak map edilir
}


class Track:
    def __init__(self, tid: int, pos_xy, t_sec: float):
        self.id = tid
        self.pos = np.array(pos_xy, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.prev_range = float(np.linalg.norm(self.pos))
        self.last_pos = self.pos.copy()
        self.last_t = t_sec

        self.age = 1
        self.miss = 0

        self.class_id = 0   # ← Environment format: integer (0-4)
        self.yolo_conf = 0.0
        self.radar_conf = 0.0
        self.intensity = 0.0

    def update(self, pos_xy, t_sec: float,
               class_id=None, yolo_conf=None,
               radar_conf=None, intensity=None):
        pos = np.array(pos_xy, dtype=np.float32)
        dt = (t_sec - self.last_t)

        if dt < 0.01 or dt > 1.0:
            self.vel *= 0.9
        else:
            dp = pos - self.last_pos
            dist_moved = np.linalg.norm(dp)

            if dist_moved < 0.25:
                self.vel[:] = 0.0
            else:
                v_inst = dp / dt
                if np.linalg.norm(v_inst) < 15.0:
                    self.vel = 0.9 * self.vel + 0.1 * v_inst

        self.last_pos = pos
        self.pos = pos
        self.last_t = t_sec

        self.age += 1
        self.miss = 0

        if class_id is not None:
            self.class_id = int(class_id)  # ← Integer olarak sakla
        if yolo_conf is not None:
            self.yolo_conf = float(yolo_conf)
        if radar_conf is not None:
            self.radar_conf = float(radar_conf)
        if intensity is not None:
            self.intensity = float(intensity)

    def step_miss(self):
        self.miss += 1


class ThreatEncoderNode(Node):
    def __init__(self):
        super().__init__("threat_encoder_node")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.base_frame = "base_link"
        self.radar_frame_default = "radar_link"
        self.gate_dist = 2.0
        self.cluster_thr = 0.6
        self.track_match_dist = 2.0
        self.max_miss = 10
        self.K = 5
        self.token_len = 17

        self.latest_yolo = None
        self.latest_radar = None
        self.tracks = {}
        self.next_id = 1

        self.create_subscription(
            Detection3DArray, "/yolo/projected_detections", self.cb_yolo, 10)
        self.create_subscription(
            RadarPoints, "/radar/points_filtered_radarmsg", self.cb_radar, 10)

        self.pub_state = self.create_publisher(Float32MultiArray, "/threat/state_vec", 10)
        self.pub_debug = self.create_publisher(String, "/threat/debug_topk", 10)

        self.uav_speed = 0.0
        self.uav_yaw = 0.0
        self.create_subscription(Odometry, "/odometry/filtered", self.cb_odom, 10)

        self.create_timer(0.1, self.tick)
        self.get_logger().info("ThreatEncoderNode started (FIXED class mapping)")

    def cb_odom(self, msg: Odometry):
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self.uav_speed = math.sqrt(vx*vx + vy*vy)

        qx = float(msg.pose.pose.orientation.x)
        qy = float(msg.pose.pose.orientation.y)
        qz = float(msg.pose.pose.orientation.z)
        qw = float(msg.pose.pose.orientation.w)
        self.uav_yaw = math.atan2(
            2.0 * (qw*qz + qx*qy),
            1.0 - 2.0 * (qy*qy + qz*qz)
        )

    def cb_yolo(self, msg: Detection3DArray):
        self.latest_yolo = msg

    def cb_radar(self, msg: RadarPoints):
        self.latest_radar = msg

    def _stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _cluster_radar(self, radar_xyi):
        if len(radar_xyi) == 0:
            return []

        pts = np.array([[p[0], p[1]] for p in radar_xyi], dtype=np.float32)
        intens = np.array([p[2] for p in radar_xyi], dtype=np.float32)

        used = np.zeros(len(pts), dtype=bool)
        clusters = []

        for i in range(len(pts)):
            if used[i]:
                continue
            group = [i]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(len(pts)):
                    if used[j]:
                        continue
                    d = np.min(np.linalg.norm(pts[group] - pts[j], axis=1))
                    if d < self.cluster_thr:
                        used[j] = True
                        group.append(j)
                        changed = True

            c = pts[group].mean(axis=0)
            inten = float(intens[group].mean())
            radar_conf = float(np.clip(inten, 0.0, 1.0))

            clusters.append({
                "pos": (float(c[0]), float(c[1])),
                "intensity": inten,
                "radar_conf": radar_conf
            })

        return clusters

    def _associate_yolo_radar(self, yolo_list, radar_clusters):
        matches = []
        used_r = set()

        for yd in yolo_list:
            best = None
            best_d = 1e9
            for ri, rc in enumerate(radar_clusters):
                if ri in used_r:
                    continue
                d = np.linalg.norm(np.array(yd["pos"]) - np.array(rc["pos"]))
                if d < best_d and d < self.gate_dist:
                    best_d = d
                    best = ri

            if best is not None:
                used_r.add(best)
                matches.append((yd, radar_clusters[best]))
            else:
                matches.append((yd, None))

        for ri, rc in enumerate(radar_clusters):
            if ri not in used_r:
                matches.append((None, rc))

        return matches

    def _find_nearest_track(self, pos_xy, max_d):
        best_id = None
        best_d = 1e9
        p = np.array(pos_xy, dtype=np.float32)
        for tid, tr in self.tracks.items():
            d = float(np.linalg.norm(tr.pos - p))
            if d < best_d and d < max_d:
                best_d = d
                best_id = tid
        return best_id

    def _update_tracks(self, matches, yolo_t: float, radar_t: float):
        updated = set()

        for yd, rc in matches:
            if rc is not None:
                pos = rc["pos"]
                t_meas = radar_t
            elif yd is not None:
                pos = yd["pos"]
                t_meas = yolo_t
            else:
                continue

            tid = self._find_nearest_track(pos, self.track_match_dist)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(tid, pos, t_meas)

            class_id = yd["class_id"] if yd is not None else None
            yolo_conf = yd["score"] if yd is not None else None
            radar_conf = rc["radar_conf"] if rc is not None else None
            intensity = rc["intensity"] if rc is not None else None

            self.tracks[tid].update(
                pos_xy=pos,
                t_sec=t_meas,
                class_id=class_id,
                yolo_conf=yolo_conf,
                radar_conf=radar_conf,
                intensity=intensity
            )
            updated.add(tid)

        to_del = []
        for tid, tr in self.tracks.items():
            if tid not in updated:
                tr.step_miss()
                if tr.miss > self.max_miss:
                    to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

    def _tokenize(self, tr: Track):
        """
        ═══════════════════════════════════════════════════════════════════
        TOKEN FORMAT (17 features) — Environment Compatible
        ═══════════════════════════════════════════════════════════════════
        0:  obj_id         (track ID)
        1:  class_id       (0=Unknown, 1=Drone, 2=Bird, 3=FixedWing, 4=Person)
        2:  confidence     (max of YOLO/radar)
        3:  intensity      (radar intensity)
        4:  dist           (range to UAV)
        5:  closing_speed  (radial velocity, positive=approaching)
        6:  sin(bearing)
        7:  cos(bearing)
        8:  vx             (velocity X, base_link frame)
        9:  vy             (velocity Y, base_link frame)
        10: x              (position X, base_link frame)
        11: y              (position Y, base_link frame)
        12: radar_conf     (radar confidence)
        13: yolo_conf      (YOLO confidence)
        14: age_norm       (track age, normalized)
        15: (reserved)
        16: is_valid       (1.0 if valid track)
        ═══════════════════════════════════════════════════════════════════
        """
        x, y = float(tr.pos[0]), float(tr.pos[1])
        vx, vy = float(tr.vel[0]), float(tr.vel[1])

        r = float(np.hypot(x, y))

        if r > 1e-3:
            closing = float(-(x * vx + y * vy) / r)
            if abs(closing) < 0.2:  # Noise gate
                closing = 0.0
            sn = float(y / r)
            cs = float(x / r)
        else:
            closing = 0.0
            sn, cs = 0.0, 1.0

        conf = max(tr.yolo_conf, tr.radar_conf)
        age_norm = float(np.clip(tr.age / 50.0, 0.0, 1.0))

        token = [
            float(tr.id),          # 0: obj_id
            float(tr.class_id),    # 1: class_id (integer 0-4)
            conf,                  # 2: confidence
            float(tr.intensity),   # 3: intensity
            r,                     # 4: dist
            closing,               # 5: closing_speed
            sn,                    # 6: sin(bearing)
            cs,                    # 7: cos(bearing)
            vx,                    # 8: vx
            vy,                    # 9: vy
            x,                     # 10: x
            y,                     # 11: y
            float(tr.radar_conf),  # 12: radar_conf
            float(tr.yolo_conf),   # 13: yolo_conf
            age_norm,              # 14: age_norm
            0.0,                   # 15: reserved
            1.0                    # 16: is_valid
        ]
        return token, r, closing

    def _pre_score(self, tr: Track):
        token, r, closing = self._tokenize(tr)
        conf = max(tr.yolo_conf, tr.radar_conf)

        # Class weight: Person > Unknown > Drone/Bird
        if tr.class_id == 4:    # Person
            cw = 1.0
        elif tr.class_id == 0:  # Unknown
            cw = 0.5
        else:                   # Drone/Bird/FixedWing
            cw = 0.2

        score = (1.0 / (r + 0.5)) + max(0.0, closing) + 0.2 * conf + 0.3 * cw
        return float(score)

    def tick(self):
        if self.latest_yolo is None or self.latest_radar is None:
            return

        yolo_t = self._stamp_to_sec(self.latest_yolo.header.stamp)
        radar_t = self._stamp_to_sec(self.latest_radar.header.stamp)

        # ── YOLO Processing ──────────────────────────────────────────────
        yolo_list = []
        for d in self.latest_yolo.detections:
            if len(d.results) == 0:
                continue
            hypo = d.results[0]
            x = float(hypo.pose.pose.position.x)
            y = float(hypo.pose.pose.position.y)

            # ═══ CLASS MAPPING FIX ═══════════════════════════════════════
            yolo_class_str = str(hypo.hypothesis.class_id)
            env_class_id = YOLO_TO_ENV_CLASS.get(yolo_class_str, 0)  # Default: Unknown
            # ═════════════════════════════════════════════════════════════

            yolo_list.append({
                "pos": (x, y),
                "class_id": env_class_id,  # ← Environment-compatible class ID
                "score": float(hypo.hypothesis.score)
            })

        # ── Radar Processing ─────────────────────────────────────────────
        radar_xyi = []
        r_stamp = self.latest_radar.header.stamp
        radar_frame = self.latest_radar.header.frame_id or self.radar_frame_default

        tf_ok = True
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, radar_frame,
                rclpy.time.Time.from_msg(r_stamp)
            )
        except Exception:
            tf_ok = False
            tf = None

        for rp in self.latest_radar.points:
            x = float(rp.x)
            y = float(rp.y)
            z = float(rp.z)
            inten = float(getattr(rp, "intensity", 0.0))

            if tf_ok:
                ps = PointStamped()
                ps.header.stamp = r_stamp
                ps.header.frame_id = radar_frame
                ps.point.x = x
                ps.point.y = y
                ps.point.z = z
                try:
                    out = tf2_geometry_msgs.do_transform_point(ps, tf)
                    xb, yb = float(out.point.x), float(out.point.y)
                except Exception:
                    continue
            else:
                xb, yb = x, y

            radar_xyi.append((xb, yb, inten))

        radar_clusters = self._cluster_radar(radar_xyi)
        matches = self._associate_yolo_radar(yolo_list, radar_clusters)
        self._update_tracks(matches, yolo_t, radar_t)

        # ── Top-K Selection ──────────────────────────────────────────────
        scored = []
        for tid, tr in self.tracks.items():
            s = self._pre_score(tr)
            token, _, _ = self._tokenize(tr)
            scored.append((s, tid, token, tr))
        scored.sort(key=lambda x: x[0], reverse=True)
        topk = scored[: self.K]

        # ── Build state_vec ──────────────────────────────────────────────
        yaw_sin = math.sin(self.uav_yaw)
        yaw_cos = math.cos(self.uav_yaw)
        max_speed = 5.0
        speed_norm = float(np.clip(self.uav_speed / max_speed, 0.0, 1.0))

        vec = [speed_norm, yaw_sin, yaw_cos]  # UAV state (3 values)

        for i in range(self.K):
            if i < len(topk):
                vec.extend(topk[i][2])  # 17 values per object
            else:
                vec.extend([0.0] * self.token_len)

        msg = Float32MultiArray()
        msg.data = vec
        self.pub_state.publish(msg)

        # ── Debug Output ─────────────────────────────────────────────────
        parts = []
        class_names = {0: "Unknown", 1: "Drone", 2: "Bird", 3: "FixedWing", 4: "Person"}
        for s, tid, token, tr in topk:
            x, y = tr.pos
            r = float(np.hypot(x, y))
            closing = float(token[5])
            conf = max(tr.yolo_conf, tr.radar_conf)
            cls_name = class_names.get(tr.class_id, "?")
            parts.append(
                f"id={tid} cls={cls_name} r={r:.2f} "
                f"vel=({tr.vel[0]:.2f},{tr.vel[1]:.2f}) "
                f"closing={closing:.2f} conf={conf:.2f}"
            )
        dbg = String()
        dbg.data = " | ".join(parts) if parts else "no_tracks"
        self.pub_debug.publish(dbg)


def main():
    rclpy.init()
    node = ThreatEncoderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()