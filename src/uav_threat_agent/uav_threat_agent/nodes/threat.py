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

# Radar msg (senin paket)
from fusion_msgs.msg import RadarPoints
from nav_msgs.msg import Odometry
import math


# ----------------------------
# Track structure
# ----------------------------
class Track:
    def __init__(self, tid: int, pos_xy, t_sec: float):
        self.id = tid
        # Internally track full 3D state in base_link.
        # NOTE: We keep the published RL token length unchanged (still mostly 2D),
        # but scoring will use 3D range/closing + a vertical-separation penalty.
        self.pos = np.array(pos_xy, dtype=np.float32)  # (x,y,z)
        self.vel = np.zeros(3, dtype=np.float32)       # (vx,vy,vz)
        self.prev_range = float(np.linalg.norm(self.pos))
        self.last_pos = self.pos.copy()
        self.last_t = t_sec

        self.age = 1
        self.miss = 0

        self.class_id = "-1"   # string (vision_msgs uses string)
        self.yolo_conf = 0.0
        self.radar_conf = 0.0
        self.intensity = 0.0
        

    def update(self, pos_xy, t_sec: float,
               class_id=None, yolo_conf=None,
               radar_conf=None, intensity=None):
        pos = np.array(pos_xy, dtype=np.float32)  # (x,y,z)
        dt = (t_sec - self.last_t)

        # dt çok küçükse (aynı timestamp), velocity update yapma → spike önler
        if dt < 0.01 or dt > 1.0:
            # küçük bir decay yeterli
            self.vel *= 0.9
        else:
            dp = pos - self.last_pos
            dist_moved = float(np.linalg.norm(dp))

            # --- FIX STARTS HERE ---
            
            # 1. Zero-Velocity Gate: Increase threshold to 20-25cm
            # If it moved less than this, FORCE velocity to zero.
            if dist_moved < 0.25: 
                self.vel[:] = 0.0
            else:
                # 2. Instantaneous velocity
                v_inst = dp / dt
                
                # 3. Cap maximum realistic acceleration/velocity (Sanity Check)
                # If a ball suddenly "moves" at 50 m/s, it's a glitch.
                if float(np.linalg.norm(v_inst)) < 15.0: 
                    # 4. Smoother Alpha: Trust history (0.9) more than new noisy data (0.1)
                    self.vel = 0.9 * self.vel + 0.1 * v_inst
                

        self.last_pos = pos
        self.pos = pos
        self.last_t = t_sec

        self.age += 1
        self.miss = 0

        if class_id is not None:
            self.class_id = str(class_id)
        if yolo_conf is not None:
            self.yolo_conf = float(yolo_conf)
        if radar_conf is not None:
            self.radar_conf = float(radar_conf)
        if intensity is not None:
            self.intensity = float(intensity)

    def step_miss(self):
        self.miss += 1


# ----------------------------
# Main encoder node
# ----------------------------
class ThreatEncoderNode(Node):
    def __init__(self):
        super().__init__("threat_encoder_node")

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Params (istersen ros param yaparız)
        self.base_frame = "base_link"
        self.radar_frame_default = "radar_link"  # msg.header.frame_id ile override olur
        self.gate_dist = 2.0          # YOLO-radar association gate (m)
        self.cluster_thr = 0.6        # radar clustering distance (m)
        self.track_match_dist = 2.0   # track update gate (m)
        self.max_miss = 10            # kaç tick kaybolunca silinsin
        self.K = 5                    # Top-K
        self.token_len = 17           # aşağıdaki tokenize() ile uyumlu

        # Vertical separation handling:
        # If a target is at same XY but far above/below, it should be less threatening.
        # z is in base_link (relative). We'll down-weight score using these parameters.
        self.z_soft_m = 1.0    # start penalizing above this |z|
        self.z_hard_m = 3.0    # strong penalty above this |z|
        self.z_min_penalty = 0.05  # never go below this multiplier

        # Buffers
        self.latest_yolo = None
        self.latest_radar = None

        # Tracks
        self.tracks = {}
        self.next_id = 1
        

        # Subs
        self.create_subscription(
            Detection3DArray,
            "/yolo/projected_detections",
            self.cb_yolo,
            10
        )
        self.create_subscription(
            RadarPoints,
            "/radar/points_filtered_radarmsg",
            self.cb_radar,
            10
        )

        # Pubs
        self.pub_state = self.create_publisher(Float32MultiArray, "/threat/state_vec", 10)
        self.pub_debug = self.create_publisher(String, "/threat/debug_topk", 10)

        # Timer: encoder tick (ör. 10 Hz)
        self.create_timer(0.1, self.tick)

        
        self.uav_speed = 0.0
        self.uav_yaw = 0.0
        self.create_subscription(
                Odometry,
                "/odometry/filtered",
                self.cb_odom,
                10
            )
        self.get_logger().info("ThreatEncoderNode started.")
        
    def cb_odom(self, msg: Odometry):
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        # 2D speed
        self.uav_speed = math.sqrt(vx*vx + vy*vy)

        # quaternion -> yaw
        qx = float(msg.pose.pose.orientation.x)
        qy = float(msg.pose.pose.orientation.y)
        qz = float(msg.pose.pose.orientation.z)
        qw = float(msg.pose.pose.orientation.w)

        self.uav_yaw = math.atan2(
            2.0 * (qw*qz + qx*qy),
            1.0 - 2.0 * (qy*qy + qz*qz)
        )

    # ---------- callbacks ----------
    def cb_yolo(self, msg: Detection3DArray):
        self.latest_yolo = msg

    def cb_radar(self, msg: RadarPoints):
        self.latest_radar = msg

    # ---------- helpers ----------
    def _stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _tf_point(self, x, y, z, stamp_ros, src_frame: str, dst_frame: str):
        ps = PointStamped()
        ps.header.stamp = stamp_ros
        ps.header.frame_id = src_frame
        ps.point.x = float(x)
        ps.point.y = float(y)
        ps.point.z = float(z)

        tf = self.tf_buffer.lookup_transform(
            dst_frame,
            src_frame,
            rclpy.time.Time.from_msg(stamp_ros)
        )
        out = tf2_geometry_msgs.do_transform_point(ps, tf)
        return out.point.x, out.point.y, out.point.z

    def _cluster_radar(self, radar_xyi):
        """
        radar_xyi: list of (x,y,z,intensity)
        naive proximity clustering in 3D: grow groups with cluster_thr
        returns list of dict {pos:(x,y,z), intensity, radar_conf}
        """
        if len(radar_xyi) == 0:
            return []

        pts = np.array([[p[0], p[1], p[2]] for p in radar_xyi], dtype=np.float32)
        intens = np.array([p[3] for p in radar_xyi], dtype=np.float32)

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
                    d = float(np.min(np.linalg.norm(pts[group] - pts[j], axis=1)))
                    if d < self.cluster_thr:
                        used[j] = True
                        group.append(j)
                        changed = True

            c = pts[group].mean(axis=0)
            inten = float(intens[group].mean())

            # radar_conf: intensity bazlı kaba normalizasyon (0..1 zaten gibi)
            radar_conf = float(np.clip(inten, 0.0, 1.0))

            clusters.append({
                "pos": (float(c[0]), float(c[1]), float(c[2])),
                "intensity": inten,
                "radar_conf": radar_conf
            })

        return clusters

    def _associate_yolo_radar(self, yolo_list, radar_clusters):
        """
        yolo_list: list of dict {pos:(x,y,0), class_id, score}
        radar_clusters: list of dict {pos:(x,y,z), intensity, radar_conf}
        returns list of tuples: (yolo or None, radar or None)
        """
        matches = []
        used_r = set()

        # each yolo -> nearest radar
        for yd in yolo_list:
            best = None
            best_d = 1e9
            for ri, rc in enumerate(radar_clusters):
                if ri in used_r:
                    continue
                # Prefer matching objects near the ground plane (YOLO projection gives z=0).
                # If radar z is large, it likely isn't the same object.
                z_abs = abs(float(rc["pos"][2]))
                if z_abs > self.z_hard_m:
                    continue
                d = float(np.linalg.norm(np.array(yd["pos"], dtype=np.float32) - np.array(rc["pos"], dtype=np.float32)))
                if d < best_d and d < self.gate_dist:
                    best_d = d
                    best = ri

            if best is not None:
                used_r.add(best)
                matches.append((yd, radar_clusters[best]))
            else:
                matches.append((yd, None))

        # leftover radar (no yolo)
        for ri, rc in enumerate(radar_clusters):
            if ri not in used_r:
                matches.append((None, rc))

        return matches

    def _find_nearest_track(self, pos_xy, max_d):
        best_id = None
        best_d = 1e9
        p = np.array(pos_xy, dtype=np.float32)  # (x,y,z)
        for tid, tr in self.tracks.items():
            d = float(np.linalg.norm(tr.pos - p))
            if d < best_d and d < max_d:
                best_d = d
                best_id = tid
        return best_id

    def _update_tracks(self, matches, yolo_t: float, radar_t: float):
        updated = set()

        for yd, rc in matches:
            # pos + timestamp seçimi:
            # radar varsa radar zamanını kullan (daha stabil)
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

        # miss + delete
        to_del = []
        for tid, tr in self.tracks.items():
            if tid not in updated:
                tr.step_miss()
                if tr.miss > self.max_miss:
                    to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

    def _class_to_onehot4(self, class_id_str: str):
        """
        Sen sonra burada gerçek mapping yapacaksın.
        Şimdilik: her şey unknown.
        Format: [unknown, human, vehicle, obstacle]
        """
        onehot = [1.0, 0.0, 0.0, 0.0]

        # İstersen hızlı demo mapping:
        # COCO person=0
        if class_id_str == "0":
            onehot = [0.0, 1.0, 0.0, 0.0]

        return onehot

    def _tokenize(self, tr: Track):
        # Keep token shape unchanged (2D fields), but compute range/closing using 3D.
        x, y, z = float(tr.pos[0]), float(tr.pos[1]), float(tr.pos[2])
        vx, vy, vz = float(tr.vel[0]), float(tr.vel[1]), float(tr.vel[2])

        r_xy = float(np.hypot(x, y))
        r_3d = float(np.linalg.norm(tr.pos))

        # radial closing rate in 3D (positive approaching)
        if r_3d > 1e-3:
            closing = float(-(x * vx + y * vy + z * vz) / r_3d)
            # Direction encoding stays horizontal (2D bearing)
            if r_xy > 1e-3:
                sn = float(y / r_xy)
                cs = float(x / r_xy)
            else:
                sn, cs = 0.0, 1.0
        else:
            closing = 0.0
            if abs(closing) < 0.2:
                closing = 0.0
            sn, cs = 0.0, 1.0

        onehot = self._class_to_onehot4(tr.class_id)

        # age normalize: kaba (0..1)
        age_norm = float(np.clip(tr.age / 50.0, 0.0, 1.0))

        token = [
            x, y,
            vx, vy,
            r_3d,
            closing,
            sn, cs,
            onehot[0], onehot[1], onehot[2], onehot[3],
            float(tr.radar_conf),
            float(tr.yolo_conf),
            float(tr.intensity),
            age_norm,
            1.0  # is_valid
        ]
        return token, r_3d, closing

    def _pre_score(self, tr: Track):
        token, r, closing = self._tokenize(tr)

        conf = max(tr.yolo_conf, tr.radar_conf)
        # class weight (şimdilik person yüksek)
        if tr.class_id == "0":
            cw = 1.0
        else:
            cw = 0.3 if tr.class_id != "-1" else 0.1

        # Vertical-separation penalty (in base_link).
        z_abs = abs(float(tr.pos[2]))
        if z_abs <= self.z_soft_m:
            z_pen = 1.0
        elif z_abs >= self.z_hard_m:
            z_pen = self.z_min_penalty
        else:
            # smooth transition between soft/hard
            t = (z_abs - self.z_soft_m) / max(1e-6, (self.z_hard_m - self.z_soft_m))
            z_pen = (1.0 - t) + t * self.z_min_penalty

        score = ((1.0 / (r + 0.5)) + max(0.0, closing) + 0.2 * conf + 0.3 * cw) * float(z_pen)
        return float(score)

    # ---------- main tick ----------
    def tick(self):
        if self.latest_yolo is None or self.latest_radar is None:
            return

        # Use a "now" time based on latest yolo stamp (reasonable)
        yolo_t = self._stamp_to_sec(self.latest_yolo.header.stamp)
        radar_t = self._stamp_to_sec(self.latest_radar.header.stamp)

        # ---- YOLO list (already base_link) ----
        yolo_list = []
        for d in self.latest_yolo.detections:
            if len(d.results) == 0:
                continue
            hypo = d.results[0]
            x = float(hypo.pose.pose.position.x)
            y = float(hypo.pose.pose.position.y)
            yolo_list.append({
                "pos": (x, y, 0.0),
                "class_id": hypo.hypothesis.class_id,
                "score": float(hypo.hypothesis.score)
            })

        # ---- Radar points -> base_link ----
        radar_xyi = []  # (x,y,z,intensity) in base_link
        r_stamp = self.latest_radar.header.stamp
        radar_frame = self.latest_radar.header.frame_id or self.radar_frame_default

        # TF lookup once for speed (use stamp if possible)
        tf_ok = True
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                radar_frame,
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
                    xb, yb, zb = float(out.point.x), float(out.point.y), float(out.point.z)
                except Exception:
                    continue
            else:
                # fallback: assume already base_link (not ideal)
                xb, yb, zb = x, y, z

            radar_xyi.append((xb, yb, zb, inten))

        # ---- Radar clustering ----
        radar_clusters = self._cluster_radar(radar_xyi)

        # ---- Association YOLO <-> Radar ----
        matches = self._associate_yolo_radar(yolo_list, radar_clusters)

        # ---- Track update ----
        self._update_tracks(matches, yolo_t, radar_t)

        # ---- Top-K select ----
        scored = []
        for tid, tr in self.tracks.items():
            s = self._pre_score(tr)
            token, _, _ = self._tokenize(tr)
            scored.append((s, tid, token, tr))
        scored.sort(key=lambda x: x[0], reverse=True)
        topk = scored[: self.K]

        # ---- Build state_vec ----
        # UAV state (şimdilik 0): speed, yaw_sin, yaw_cos
        yaw_sin = math.sin(self.uav_yaw)
        yaw_cos = math.cos(self.uav_yaw)

        # normalize: max_speed = 5 m/s (istersen 10 yap)
        max_speed = 5.0
        speed_norm = float(np.clip(self.uav_speed / max_speed, 0.0, 1.0))

        vec = [speed_norm, yaw_sin, yaw_cos]

        for i in range(self.K):
            if i < len(topk):
                vec.extend(topk[i][2])
            else:
                vec.extend([0.0] * self.token_len)

        msg = Float32MultiArray()
        msg.data = vec
        self.pub_state.publish(msg)

        # ---- Debug string ----
        # format: id:r:closing:class:conf
        parts = []
        for s, tid, token, tr in topk:
            x, y, z = tr.pos
            r = float(np.hypot(x, y))
            # closing from tokenize (token[5])
            closing = float(token[5])
            conf = max(tr.yolo_conf, tr.radar_conf)
            parts.append(f"id={tid} r={r:.2f} cl={closing:.2f} vx={tr.vel[0]:.2f} vy={tr.vel[1]:.2f} cls={tr.class_id} conf={conf:.2f}\n")
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