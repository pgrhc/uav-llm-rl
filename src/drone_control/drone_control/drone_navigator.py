#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math
import time
import subprocess

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleLocalPosition

from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

# ─── AYARLAR ──────────────────────────────────────────────────────────────────
LOOKAHEAD_DIST  = 2.0
WAYPOINT_RADIUS = 1.5
YAW_DEADBAND    = 0.5

CRASH_VEL_THRESH  = 0.05   # m/s — Bu hızın altında takılı kalınca
CRASH_STUCK_TIME  = 4.0    # s   — Bu kadar hareketsiz kalırsa kaza say
MISSION_ALT_M     = 1.8    # m   — Hedef irtifa

# Gazebo'daki drone model adı (gz model --list ile kontrol et)
DRONE_MODEL_NAME  = "x500_mono_cam_0"
# Drone spawn noktası — maze_with_dynamic_obstacles.py ile aynı
SPAWN_X_ENU =  0.0   # ENU X (metre)
SPAWN_Y_ENU =  0.0   # ENU Y (metre)
SPAWN_Z_ENU =  1.8   # ENU Z — irtifada başlasın
WORLD_NAME   = "default"
# ─────────────────────────────────────────────────────────────────────────────


class OffboardControl(Node):

    def __init__(self):
        super().__init__('offboard_control')

        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_std = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_px4)
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_px4)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_px4)
        self.crash_pub = self.create_publisher(Bool, '/crash_detected', 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_cb, qos_std)
        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_cb, qos_std)
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self.local_pos_cb, qos_px4)

        self.offboard_setpoint_counter = 0
        self.current_path   = []
        self.wp_index       = 0
        self.pos_enu        = [0.0, 0.0, 0.0]
        self.mission_alt    = -MISSION_ALT_M
        self.last_valid_yaw = 0.0

        # Kaza durumu
        self.local_vx           = 0.0
        self.local_vy           = 0.0
        self.local_z            = -MISSION_ALT_M
        self.stuck_since        = None
        self.crash_detected     = False
        self.crash_recovery_until = 0.0

        self.timer = self.create_timer(0.1, self.timer_cb)
        self.get_logger().info("Offboard Node başlatıldı (Kaza Koruması + Gazebo Reset Aktif)")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def odom_cb(self, msg):
        self.pos_enu = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]

    def path_cb(self, msg):
        if self.crash_detected:
            return
        self.current_path = msg.poses
        self.wp_index     = 0
        self.get_logger().info(f"Yeni rota: {len(self.current_path)} waypoint")

    def local_pos_cb(self, msg: VehicleLocalPosition):
        self.local_vx = msg.vx
        self.local_vy = msg.vy
        self.local_z  = msg.z

    # ── Timer ─────────────────────────────────────────────────────────────────

    def timer_cb(self):
        if self.offboard_setpoint_counter == 10:
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
            self.arm()

        self._check_crash()
        self.publish_offboard_control_mode()

        if self.crash_detected:
            self._do_recovery()
        else:
            self.publish_trajectory_setpoint()

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

    # ── Kaza Tespiti ──────────────────────────────────────────────────────────

    def _check_crash(self):
        if self.crash_detected:
            if time.time() > self.crash_recovery_until:
                self.get_logger().info("✅ Recovery tamamlandı.")
                self.crash_detected = False
                self.stuck_since    = None
                # Path yeniden gelecek (auto_maze_navigator gönderecek)
            return

        now = time.time()

        # 1. Yere çarptı mı?
        if self.local_z > -0.3:
            self._trigger_crash("Yere çarptı (z={:.2f})".format(self.local_z))
            return

        # 2. Takıldı mı?
        if self.current_path and self.wp_index < len(self.current_path):
            speed = math.hypot(self.local_vx, self.local_vy)
            if speed < CRASH_VEL_THRESH:
                if self.stuck_since is None:
                    self.stuck_since = now
                elif now - self.stuck_since > CRASH_STUCK_TIME:
                    self._trigger_crash(f"{CRASH_STUCK_TIME}s hareketsiz kaldı")
                    return
            else:
                self.stuck_since = None

    def _trigger_crash(self, reason: str):
        self.get_logger().error(f"🚨 KAZA: {reason}")
        self.crash_detected = True

        # 1. Environment'a bildir
        msg = Bool()
        msg.data = True
        self.crash_pub.publish(msg)

        # 2. Drone'u Gazebo'da spawn noktasına taşı
        self._reset_drone_in_gazebo()

        # 3. PX4'ü yeniden arm et (kaza sonrası disarm olabilir)
        self.crash_recovery_until = time.time() + 4.0   # 4s bekle

        # 4. Path temizle
        self.current_path = []
        self.wp_index     = 0

    def _reset_drone_in_gazebo(self):
        """
        Drone'u Gazebo'da spawn noktasına ışınla.
        ENU → Gazebo koordinatı (Gazebo X=East, Y=North, Z=Up → ENU ile aynı)
        """
        gz_x = SPAWN_X_ENU
        gz_y = SPAWN_Y_ENU
        gz_z = SPAWN_Z_ENU

        # pose string: "x y z roll pitch yaw"
        pose_str = f"{gz_x} {gz_y} {gz_z} 0 0 0"

        cmd = (
            f"gz service -s /world/{WORLD_NAME}/set_pose "
            f"--reqtype gz.msgs.Pose "
            f"--reptype gz.msgs.Boolean "
            f"--timeout 2000 "
            f"--req 'name: \"{DRONE_MODEL_NAME}\" "
            f"position: {{x: {gz_x}, y: {gz_y}, z: {gz_z}}} "
            f"orientation: {{x: 0, y: 0, z: 0, w: 1}}'"
        )

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True,
                                    text=True, timeout=3.0)
            if "true" in result.stdout.lower():
                self.get_logger().info(
                    f"🔄 Drone spawn noktasına taşındı: ({gz_x}, {gz_y}, {gz_z})"
                )
            else:
                self.get_logger().warn(
                    f"Gazebo reset yanıt vermedi: {result.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            self.get_logger().warn("Gazebo reset timeout!")
        except Exception as e:
            self.get_logger().warn(f"Gazebo reset hatası: {e}")

    def _do_recovery(self):
        """Kaza sonrası drone'u hedef irtifada tut."""
        msg = TrajectorySetpoint()
        msg.position  = [self.pos_enu[1], self.pos_enu[0], self.mission_alt]
        msg.velocity  = [float('nan'), float('nan'), float('nan')]
        msg.yaw       = self.last_valid_yaw
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.trajectory_setpoint_pub.publish(msg)

        # 4s sonunda yeniden arm et
        remaining = self.crash_recovery_until - time.time()
        if 0.0 < remaining < 0.5:
            self.get_logger().info("Yeniden arm ediliyor...")
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
            self.arm()

    # ── Lookahead ─────────────────────────────────────────────────────────────

    def _advance_wp_index(self):
        cx, cy = self.pos_enu[0], self.pos_enu[1]
        while self.wp_index < len(self.current_path):
            wp   = self.current_path[self.wp_index].pose.position
            dist = math.hypot(wp.x - cx, wp.y - cy)
            if dist <= WAYPOINT_RADIUS:
                self.wp_index += 1
                self.get_logger().info(f"✅ WP {self.wp_index}/{len(self.current_path)}")
            else:
                break

    def _get_lookahead(self):
        if not self.current_path or self.wp_index >= len(self.current_path):
            return None
        cx, cy = self.pos_enu[0], self.pos_enu[1]
        for i in range(self.wp_index, len(self.current_path)):
            wp   = self.current_path[i].pose.position
            dist = math.hypot(wp.x - cx, wp.y - cy)
            if dist >= LOOKAHEAD_DIST:
                return wp
        return self.current_path[-1].pose.position

    # ── Setpoint ──────────────────────────────────────────────────────────────

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        self._advance_wp_index()
        target = self._get_lookahead()

        if target is not None:
            t_north = target.y
            t_east  = target.x
            dn = t_north - self.pos_enu[1]
            de = t_east  - self.pos_enu[0]
            dist = math.hypot(dn, de)

            msg.position = [t_north, t_east, self.mission_alt]
            msg.velocity = [float('nan'), float('nan'), float('nan')]

            if dist > YAW_DEADBAND:
                self.last_valid_yaw = math.atan2(de, dn)
            msg.yaw = self.last_valid_yaw

            if self.offboard_setpoint_counter % 20 == 0:
                self.get_logger().info(
                    f"WP {self.wp_index}/{len(self.current_path)} | dist={dist:.1f}m"
                )
        else:
            msg.position = [self.pos_enu[1], self.pos_enu[0], self.mission_alt]
            msg.velocity = [float('nan'), float('nan'), float('nan')]
            msg.yaw      = self.last_valid_yaw
            if self.offboard_setpoint_counter % 30 == 0:
                self.get_logger().info("Path bekleniyor (hover)")

        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.trajectory_setpoint_pub.publish(msg)

    # ── OffboardControlMode ───────────────────────────────────────────────────

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position  = True
        msg.velocity  = False
        msg.acceleration = False
        msg.attitude  = False
        msg.body_rate = False
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.offboard_control_mode_pub.publish(msg)

    # ── Komutlar ──────────────────────────────────────────────────────────────

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.param1 = param1;  msg.param2 = param2
        msg.command = command
        msg.target_system = 1;    msg.target_component = 1
        msg.source_system = 1;    msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
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