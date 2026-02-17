#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand

from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


# ─── HIZLANMA PARAMETRESİ ───────────────────────────────────────────────────
# Simülasyonda test et, gerçek uçuşta 1.5-2.0 ile başla
MAX_SPEED = 1.5   # m/s   ← İstediğin kadar artırabilirsin (simülasyon: 5.0'e kadar)


class OffboardControl(Node):

    def __init__(self):
        super().__init__('offboard_control')

        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        qos_standard = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- Publisherlar ---
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_px4)

        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_px4)

        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_px4)

        # --- Subscriberlar ---
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, qos_standard)

        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_callback, qos_standard)

        self.route_waypoint_sub = self.create_subscription(
            PoseStamped, '/route/waypoint_safe', self.route_waypoint_callback, 10)

        # --- Değişkenler ---
        self.offboard_setpoint_counter = 0
        self.current_path    = []
        self.current_wp_index = 0
        self.current_pos_enu  = [0.0, 0.0, 0.0]
        self.mission_altitude = -1.8
        self.acceptance_radius = 2.5

        self.route_waypoint      = None
        self.route_waypoint_timeout = 1.0
        self.route_waypoint_time = None

        self.log_counter = 0

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info(
            f"Offboard Node Başlatıldı. MAX_SPEED={MAX_SPEED} m/s. "
            "Path/Route waypoint bekleniyor..."
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def odom_callback(self, msg):
        self.current_pos_enu = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]

    def path_callback(self, msg):
        self.current_path     = msg.poses
        self.current_wp_index = 0
        self.get_logger().info(
            f"!!! YENİ ROTA ALINDI !!! Uzunluk: {len(self.current_path)} nokta."
        )

    def route_waypoint_callback(self, msg: PoseStamped):
        self.route_waypoint      = msg
        self.route_waypoint_time = self.get_clock().now()

    def is_route_waypoint_valid(self) -> bool:
        if self.route_waypoint is None or self.route_waypoint_time is None:
            return False
        elapsed = (self.get_clock().now() - self.route_waypoint_time).nanoseconds / 1e9
        return elapsed < self.route_waypoint_timeout

    # ── Timer ─────────────────────────────────────────────────────────────────

    def timer_callback(self):
        self.log_counter += 1

        if self.offboard_setpoint_counter == 10:
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
            self.arm()

        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

    # ── Setpoint Hesabı ───────────────────────────────────────────────────────

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()

        if self.current_path and self.current_wp_index < len(self.current_path):

            target = self.current_path[self.current_wp_index].pose.position

            # ENU → NED dönüşümü
            px4_north = target.y
            px4_east  = target.x
            px4_down  = self.mission_altitude

            curr_north = self.current_pos_enu[1]
            curr_east  = self.current_pos_enu[0]

            delta_north = px4_north - curr_north
            delta_east  = px4_east  - curr_east
            dist_2d = math.sqrt(delta_north**2 + delta_east**2)

            # ─── HIZLANDIRMA: Velocity Feedforward ─────────────────────────
            # OffboardControlMode'da velocity=True olduğundan PX4 bu komutu dinler.
            # Hedefe olan yönde MAX_SPEED ile uçar, yaklaşınca yavaşlar.
            if dist_2d > 0.1:
                # Normalize et, sonra MAX_SPEED ile ölçekle
                # (dist_2d'den büyük olamaz → clamp)
                scale = min(MAX_SPEED, dist_2d * 2.0) / dist_2d
                vn = delta_north * scale
                ve = delta_east  * scale
                # Hedefe 1m'den yakınsa yavaşla (yumuşak durma)
                if dist_2d < 1.0:
                    slow_factor = dist_2d   # 0-1 arası
                    vn *= slow_factor
                    ve *= slow_factor
            else:
                vn, ve = 0.0, 0.0
            # ───────────────────────────────────────────────────────────────

            msg.position = [px4_north, px4_east, px4_down]
            msg.velocity = [vn, ve, 0.0]   # ← feedforward hız

            if dist_2d > 0.1:
                msg.yaw = math.atan2(delta_east, delta_north)
            else:
                msg.yaw = float('nan')

            if self.log_counter % 20 == 0:
                self.get_logger().info(
                    f"Gidiliyor → WP:{self.current_wp_index}/{len(self.current_path)} "
                    f"Dist:{dist_2d:.2f}m "
                    f"Vel:[{vn:.1f}, {ve:.1f}] m/s "
                    f"Hedef(NED):[{px4_north:.2f}, {px4_east:.2f}]"
                )

            if dist_2d < self.acceptance_radius:
                self.current_wp_index += 1
                self.get_logger().info(
                    f"*** Waypoint {self.current_wp_index} Ulaşıldı! ***"
                )

        else:
            # Path yok → havada bekle
            hold_north = self.current_pos_enu[1]
            hold_east  = self.current_pos_enu[0]
            msg.position = [hold_north, hold_east, self.mission_altitude]
            msg.velocity = [0.0, 0.0, 0.0]
            msg.yaw      = float('nan')

            if self.log_counter % 30 == 0:
                self.get_logger().info("Path bekleniyor... (Havada sabit)")

        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.trajectory_setpoint_pub.publish(msg)

    # ── OffboardControlMode ───────────────────────────────────────────────────

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position     = True
        msg.velocity     = True    # ← AÇILDI: PX4 velocity setpoint'i de dinlesin
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        msg.timestamp    = self.get_clock().now().nanoseconds // 1000
        self.offboard_control_mode_pub.publish(msg)

    # ── Komutlar ──────────────────────────────────────────────────────────────

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.param1           = param1
        msg.param2           = param2
        msg.command          = command
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        msg.timestamp        = self.get_clock().now().nanoseconds // 1000
        self.vehicle_command_pub.publish(msg)

    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("Arm komutu gönderildi")


def main(args=None):
    rclpy.init(args=args)
    node = OffboardControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()