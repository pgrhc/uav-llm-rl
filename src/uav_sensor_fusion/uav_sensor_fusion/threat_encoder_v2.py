#!/home/ubuntu/Desktop/ros2_env/bin/python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import LaserScan

from vision_msgs.msg import Detection3DArray

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

import numpy as np
from collections import deque

from fusion_msgs.msg import RadarPoints
from nav_msgs.msg import Odometry
import math
from sklearn.cluster import DBSCAN


YOLO_TO_ENV_CLASS = {
    "0":  4,   
    "16": 2,      
    "5":  1,   
    
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

        self.class_id = 0   
        self.yolo_conf = 0.0
        self.radar_conf = 0.0
        self.intensity = 0.0

    def update(self, pos_xy, t_sec: float,
               class_id=None, yolo_conf=None,
               radar_conf=None, intensity=None):
        pos = np.array(pos_xy, dtype=np.float32)
        dt = (t_sec - self.last_t)

        if dt < 0.01 or dt > 0.5:
            self.vel *= 0.95
        else:
            dp = pos - self.last_pos
            dist_moved = np.linalg.norm(dp)

            if dist_moved < 0.01:
                self.vel[:] *= 0.8
            else:
                v_inst = dp / dt
                alpha = 0.3 if dist_moved > 0.1 else 0.15  
                self.vel = (1 - alpha) * self.vel + alpha * v_inst

        self.last_pos = pos
        self.pos = pos
        self.last_t = t_sec

        self.age += 1
        self.miss = 0

        if class_id is not None:
            self.class_id = int(class_id) 
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
        super().__init__("threat_encoder_v2")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.base_frame = "base_link"
        self.radar_frame_default = "radar_link"
        self.gate_dist = 2.0
        self.cluster_thr = 0.6
        self.track_match_dist = 2.0
        self.max_miss = 15
        self.K = 5
        self.token_len = 7  
        self.lidar_sectors = 36 
        self.lidar_max_range = 30.0  
        
        self.latest_yolo = None
        self.latest_radar = None
        self.latest_lidar = None 
        
        self.tracks = {}
        self.next_id = 1


        self.create_subscription(
            Detection3DArray, "/yolo/projected_detections", self.cb_yolo, 10)
        self.create_subscription(
            RadarPoints, "/radar/points_filtered_radarmsg", self.cb_radar, 10)
        self.create_subscription(
            LaserScan, "/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan", self.cb_lidar, 10) 

        self.pub_state = self.create_publisher(Float32MultiArray, "/threat/state_vec", 10)
        self.pub_debug = self.create_publisher(String, "/threat/debug_topk", 10)

        self.uav_speed = 0.0
        self.uav_yaw = 0.0
        self.create_subscription(Odometry, "/odometry/filtered", self.cb_odom, 10)

        self.create_timer(0.1, self.tick)
        self.get_logger().info("ThreatEncoderNode V2 started (74-element state, 7-feature tokens)")

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

    def cb_lidar(self, msg: LaserScan):

        self.latest_lidar = msg

    def _process_lidar(self, scan_msg: LaserScan):
        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, self.lidar_max_range)
        ranges = np.clip(ranges, 0.0, self.lidar_max_range)
        
        num_rays = len(ranges)
        rays_per_sector = max(1, num_rays // self.lidar_sectors)
        
        sector_distances = []
        for i in range(self.lidar_sectors):
            start_idx = i * rays_per_sector
            end_idx = start_idx + rays_per_sector if i < self.lidar_sectors - 1 else num_rays
            
            sector_rays = ranges[start_idx:end_idx]
            if len(sector_rays) > 0:
                min_dist = float(np.min(sector_rays))
            else:
                min_dist = self.lidar_max_range
            
            normalized = float(np.clip(min_dist / self.lidar_max_range, 0.0, 1.0))
            sector_distances.append(normalized)
        
        return sector_distances

    def _stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _cluster_radar(self, radar_xyi):
        if len(radar_xyi) == 0:
            return []

        pts = np.array([[p[0], p[1]] for p in radar_xyi], dtype=np.float32)
        intens = np.array([p[2] for p in radar_xyi], dtype=np.float32)

        eps = float(getattr(self, "dbscan_eps", 0.6))
        min_samples = int(getattr(self, "dbscan_min_samples", 4))

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)

        clusters = []
        unique_labels = [lb for lb in np.unique(labels) if lb != -1]

        if len(unique_labels) == 0:
            return []

        p10 = float(np.percentile(intens, 10))
        p90 = float(np.percentile(intens, 90))
        denom = (p90 - p10) if (p90 - p10) > 1e-6 else 1.0

        for lb in unique_labels:
            idx = np.where(labels == lb)[0]
            if idx.size == 0:
                continue

            c = pts[idx].mean(axis=0)
            inten_mean = float(intens[idx].mean())
            radar_conf = float(np.clip((inten_mean - p10) / denom, 0.0, 1.0))

            clusters.append({
                "pos": (float(c[0]), float(c[1])),
                "intensity": inten_mean,
                "radar_conf": radar_conf,
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

    def _find_nearest_track(self, pos_xy, max_d, dt_since_last=0.1):
        best_id = None
        best_d = 1e9
        p = np.array(pos_xy, dtype=np.float32)
        
        for tid, tr in self.tracks.items():
            predicted_pos = tr.pos + tr.vel * dt_since_last
            d_pos = float(np.linalg.norm(predicted_pos - p))
            
            if d_pos < best_d and d_pos < max_d:
                best_d = d_pos
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

    def _tokenize(self, tr: Track, lidar_distances):
        x, y = float(tr.pos[0]), float(tr.pos[1])
        vx, vy = float(tr.vel[0]), float(tr.vel[1])
        r = float(np.hypot(x, y))
        theta = math.atan2(y, x)
        if theta < 0:
            theta += 2 * math.pi  
            
        sector_idx = int((theta / (2 * math.pi)) * self.lidar_sectors) % self.lidar_sectors
        lidar_gercek_mesafe = lidar_distances[sector_idx] * self.lidar_max_range
        is_visible = 1.0 if r <= (lidar_gercek_mesafe + 0.5) else 0.0

        if r > 1e-3:
            closing = float(-(x * vx + y * vy) / r)
            if abs(closing) < 0.05:  
                closing = 0.0
            sn = float(y / r)
            cs = float(x / r)
        else:
            closing = 0.0
            sn, cs = 0.0, 1.0

        conf = max(tr.yolo_conf, tr.radar_conf)

        token = [
            float(tr.class_id), 
            r,              
            closing,               
            sn,                   
            cs,                   
            conf,                 
            is_visible            
        ]
        return token, r, closing, is_visible

    def _pre_score(self, tr: Track, lidar_distances):
        token, r, closing, is_visible = self._tokenize(tr, lidar_distances)
        conf = max(tr.yolo_conf, tr.radar_conf)
        if tr.class_id == 4:    
            cw = 1.0
        elif tr.class_id == 0:  
            cw = 0.5
        else:                  
            cw = 0.2

        score = 0.4 * (2.0 / (r + 0.5)) + 0.3 * max(0.0, closing) + 0.2 * conf + 0.1 * cw
        visibility_multiplier = 1.0 if is_visible > 0.5 else 0.01
        score *= visibility_multiplier
        return float(score)

    def _publish_empty(self, reason="NO_FRESH_SENSORS"):
        yaw_sin = math.sin(self.uav_yaw)
        yaw_cos = math.cos(self.uav_yaw)
        max_speed = 5.0
        speed_norm = float(np.clip(self.uav_speed / max_speed, 0.0, 1.0))

        vec = [speed_norm, yaw_sin, yaw_cos]  
        vec.extend([1.0] * self.lidar_sectors) 
        vec.extend([0.0] * (self.K * self.token_len))  

        msg = Float32MultiArray()
        msg.data = vec
        self.pub_state.publish(msg)

        dbg = String()
        dbg.data = reason
        self.pub_debug.publish(dbg)

    def tick(self):
     
        radar_msg = self.latest_radar
        yolo_msg = self.latest_yolo
        lidar_msg = self.latest_lidar

     
        self.latest_radar = None
        self.latest_yolo = None
        self.latest_lidar = None

     
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        radar_fresh = False
        yolo_fresh = False
        lidar_fresh = False

        radar_t = None
        yolo_t = None

        if radar_msg is not None:
            radar_t = self._stamp_to_sec(radar_msg.header.stamp)
            radar_fresh = (now_sec - radar_t) <= 2.0
            if not radar_fresh:
                radar_msg = None

        if yolo_msg is not None:
            yolo_t = self._stamp_to_sec(yolo_msg.header.stamp)
            yolo_fresh = (now_sec - yolo_t) <= 2.0
            if not yolo_fresh:
                yolo_msg = None

        if lidar_msg is not None:
            lidar_t = self._stamp_to_sec(lidar_msg.header.stamp)
            lidar_fresh = (now_sec - lidar_t) <= 2.0
            if not lidar_fresh:
                lidar_msg = None

       
        if radar_msg is None and yolo_msg is None and lidar_msg is None:
            self.tracks.clear()
            self._publish_empty("NO_FRESH_SENSORS")
            return

 
        lidar_distances = [1.0] * self.lidar_sectors  
        if lidar_msg is not None:
            lidar_distances = self._process_lidar(lidar_msg)

      
        if radar_t is None:
            radar_t = yolo_t
        if yolo_t is None:
            yolo_t = radar_t

    
        yolo_list = []
        if yolo_msg is not None:
            for d in yolo_msg.detections:
                if len(d.results) == 0:
                    continue
                hypo = d.results[0]
                yolo_class_str = str(hypo.hypothesis.class_id)
                env_class_id = YOLO_TO_ENV_CLASS.get(yolo_class_str, 0)
                yolo_list.append({
                    "pos": (float(hypo.pose.pose.position.x), float(hypo.pose.pose.position.y)),
                    "class_id": env_class_id,
                    "score": float(hypo.hypothesis.score)
                })

     
        radar_clusters = []
        if radar_msg is not None:
            radar_xyi = []
            r_stamp = radar_msg.header.stamp
            radar_frame = radar_msg.header.frame_id or self.radar_frame_default

            tf_ok = True
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    radar_frame,
                    rclpy.time.Time.from_msg(r_stamp)
                )
            except Exception as e:
                self.get_logger().warn(f"TF dönüşümü başarısız: {str(e)}")
                tf_ok = False

            for rp in radar_msg.points:
                x, y, z = float(rp.x), float(rp.y), float(rp.z)
                inten = float(getattr(rp, "intensity", 0.0))

                if tf_ok:
                    ps = PointStamped()
                    ps.header.stamp = r_stamp
                    ps.header.frame_id = radar_frame
                    ps.point.x, ps.point.y, ps.point.z = x, y, z
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

  
        scored = []
        for tid, tr in self.tracks.items():
            s = self._pre_score(tr, lidar_distances)
            token, _, _, _ = self._tokenize(tr, lidar_distances)
            scored.append((s, tid, token, tr))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        topk = scored[:self.K]

        
        yaw_sin = math.sin(self.uav_yaw)
        yaw_cos = math.cos(self.uav_yaw)
        max_speed = 5.0
        speed_norm = float(np.clip(self.uav_speed / max_speed, 0.0, 1.0))

        vec = [speed_norm, yaw_sin, yaw_cos]  
        vec.extend(lidar_distances)            

        
        for i in range(self.K):
            if i < len(topk):
                vec.extend(topk[i][2])  
            else:
                
                vec.extend([
                    0.0,  
                    0.0,
                    0.0,  
                    0.0,  
                    0.0,  
                    0.0,  
                    0.0 
                ])

        msg = Float32MultiArray()
        msg.data = vec
        self.pub_state.publish(msg)

       
        parts = []
        class_names = {0: "Unknown", 1: "Drone", 2: "Bird", 3: "FixedWing", 4: "Person"}
        for s, tid, token, tr in topk:
            x, y = tr.pos
            r = float(np.hypot(x, y))
            closing = float(token[2])  
            conf = float(token[5])    
            cls_name = class_names.get(tr.class_id, "?")
            parts.append(
                f"id={tid} cls={cls_name} r={r:.2f} "
                f"vel=({tr.vel[0]:.2f},{tr.vel[1]:.2f}) "
                f"closing={closing:.2f} conf={conf:.2f}"
            )
        
        dbg = String()
        dbg.data = " | ".join(parts) if parts else "Takip yok"
        self.pub_debug.publish(dbg)

def main():
    rclpy.init()
    node = ThreatEncoderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()