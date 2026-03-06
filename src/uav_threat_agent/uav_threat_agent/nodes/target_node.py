#!/usr/bin/env python3
import json
import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray, String


class ThreatTargetNode(Node):
    """
    Threat Target Calculator Node
    =============================
    Input:
      /threat/state_vec   (std_msgs/Float32MultiArray)  -> 74 eleman

    Output:
      /threat/target_scores   (std_msgs/Float32MultiArray) -> 5 elemanlı target risk score
      /threat/target_info     (std_msgs/String)            -> JSON info/debug
    """

    def __init__(self):
        super().__init__("threat_target_node")

        self.K = 5
        self.token_len = 7
        self.state_dim = 74

        self.class_seen_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.prev_scores = np.zeros(self.K, dtype=np.float32)

        # Subscriber
        self.sub_vec = self.create_subscription(
            Float32MultiArray,
            "/threat/state_vec",
            self.vec_callback,
            10
        )

        # Publishers
        self.pub_scores = self.create_publisher(
            Float32MultiArray,
            "/threat/target_scores",
            10
        )

        self.pub_info = self.create_publisher(
            String,
            "/threat/target_info",
            10
        )

        self.class_names = {
            0: "Unknown",
            1: "Drone",
            2: "Bird",
            3: "FixedWing",
            4: "Person"
        }

        self.get_logger().info("ThreatTargetNode başlatıldı.")

    def vec_callback(self, msg: Float32MultiArray):
        try:
            data = np.array(msg.data, dtype=np.float32)

            if data.shape[0] != self.state_dim:
                self.get_logger().warn(
                    f"Beklenen state dim={self.state_dim}, gelen={data.shape[0]}"
                )
                return

            target_scores, info_data = self.calculate_targets_and_info(data)

            # 1) Target scores publish
            score_msg = Float32MultiArray()
            score_msg.data = target_scores.astype(np.float32).tolist()
            self.pub_scores.publish(score_msg)

            # 2) Info publish (JSON string)
            info_msg = String()
            info_msg.data = json.dumps(info_data, ensure_ascii=False)
            self.pub_info.publish(info_msg)

            self.prev_scores = target_scores.copy()

        except Exception as e:
            self.get_logger().warn(f"vec_callback hatası: {e}")

    def calculate_targets_and_info(self, obs: np.ndarray):
        """
        Obs format:
        [0:3]   UAV state
        [3:39]  LiDAR 36
        [39:74] 5 obj x 7 feature
        """

        objects_flat = obs[39:]
        target_scores = np.zeros(self.K, dtype=np.float32)

        valid_obj_count = 0
        detailed_threats = []

        for i in range(self.K):
            start_idx = i * self.token_len
            obj_data = objects_flat[start_idx:start_idx + self.token_len]

            class_id = int(obj_data[0])
            dist = float(obj_data[1])
            closing_speed = float(obj_data[2])
            bearing_sin = float(obj_data[3])
            bearing_cos = float(obj_data[4])
            confidence = float(obj_data[5])
            is_valid = float(obj_data[6])

            # güvenlik kontrolü
            if class_id not in [0, 1, 2, 3, 4]:
                is_valid = 0.0

            if is_valid > 0.5:
                valid_obj_count += 1

                if class_id in self.class_seen_counts:
                    self.class_seen_counts[class_id] += 1

                # 1) Distance score
                if class_id == 4:  # Person
                    dist_score = 1.0 / (1.0 + np.exp(2.0 * (dist - 3.5)))
                else:
                    dist_score = 1.0 / (1.0 + np.exp(1.5 * (dist - 2.5)))

                # 2) Closing speed score
                speed_score = 0.0
                if closing_speed > 0.1:
                    speed_score = np.clip(0.3 * closing_speed, 0.0, 0.8)

                raw_risk = np.clip(dist_score + speed_score, 0.0, 1.0)

                # 3) Class-based factor
                if class_id == 0:  # Unknown
                    if closing_speed > 0.3:
                        c_factor = 0.8
                    elif closing_speed > 0.1:
                        c_factor = 0.4
                    else:
                        c_factor = 0.05
                elif class_id == 4:  # Person
                    c_factor = 0.9
                else:
                    c_factor = 0.0

                # 4) Optional confidence scaling
                # İstersen confidence'i kapatabilirsin.
                conf_factor = np.clip(confidence, 0.0, 1.0)

                target_risk = raw_risk * c_factor * conf_factor
                target_risk = float(np.clip(target_risk, 0.0, 1.0))

                target_scores[i] = target_risk

                # bearing açısı yalnızca debug için
                bearing_rad = float(np.arctan2(bearing_sin, bearing_cos))

                detailed_threats.append({
                    "slot": i,
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, "Unknown"),
                    "dist": round(dist, 3),
                    "closing_speed": round(closing_speed, 3),
                    "bearing_rad": round(bearing_rad, 3),
                    "confidence": round(confidence, 3),
                    "raw_risk": round(float(raw_risk), 3),
                    "target_risk": round(target_risk, 3),
                    "is_valid": 1
                })

            

        # Global metrics
        high_risk_mask = target_scores > 0.4
        high_risk_count = int(np.sum(high_risk_mask))

    

        info_data = {
            "target_scores": [round(float(x), 4) for x in target_scores.tolist()],
            "top_threats": detailed_threats,
           
        }

        return target_scores, info_data


def main(args=None):
    rclpy.init(args=args)
    node = ThreatTargetNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()