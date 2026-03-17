#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleStatus  # EKLENEN

from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

# ─── ADAPTİF AYARLAR ─────────────────────────────────────────────────────────
MAX_LOOKAHEAD = 1.5
MIN_LOOKAHEAD = 0.6
TURN_THRESHOLD = 0.5
YAW_DEADBAND = 0.2
# ─────────────────────────────────────────────────────────────────────────────

# ─── STABİLİTE AYARLARI (EKLENDİ) ────────────────────────────────────────────
TURN_ON_THRESHOLD  = 0.55
TURN_OFF_THRESHOLD = 0.35

SETPOINT_ALPHA = 0.15
MAX_SETPOINT_STEP = 0.25
TIMER_PERIOD = 0.05

NEAR_TARGET_DIST = 0.15
WAYPOINT_REACHED_DIST = 0.35

YAW_ERR_DEADBAND = 0.08
MAX_YAW_RATE = 0.6

CLOSEST_SEARCH_WINDOW = 50
PATH_RESET_DIST = 1.0

ROUTE_WP_TIMEOUT = 2.0



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
        # __init__ içine ekle
        self.declare_parameter('max_speed', 0.25)
        self.declare_parameter('alpha', 0.15)

        # timer_callback içinde oku
        self.MAX_SETPOINT_STEP = self.get_parameter('max_speed').value
        self.SETPOINT_ALPHA    = self.get_parameter('alpha').value
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_px4)
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_px4)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_px4)

        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, qos_standard)
        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_callback, qos_standard)
        self.route_wp_sub = self.create_subscription(
            PoseStamped, '/route/waypoint_desired', self.route_wp_callback, 10)
        
        # EKLENEN: Vehicle status subscriber
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_px4)

        self.offboard_setpoint_counter = 0
        self.current_path = []
        self.current_pos_enu = [0.0, 0.0, 0.0]
        self.current_yaw = 0.0
        self.mission_altitude = -1.2
        self.last_valid_yaw = 0.0

        self.in_turn_mode = False
        self.path_index = 0

        self.sp_n = None
        self.sp_e = None

        self.last_plan_start = None
        self.last_plan_end = None

        self.route_wp = None
        self.route_wp_stamp = 0.0

        # EKLENEN: Durum takibi için
        self.vehicle_status = None
        self.arming_state = 0
        self.nav_state = 0
        self.offboard_mode_attempted = False
        self.arm_attempted = False
        self.initialization_complete = False

        self.timer = self.create_timer(TIMER_PERIOD, self.timer_callback)
        self.get_logger().info(
            f"ADAPTİF MOD: Max={MAX_LOOKAHEAD}m, Min={MIN_LOOKAHEAD}m | "
            f"Timer={TIMER_PERIOD}s | Alpha={SETPOINT_ALPHA} | MaxStep={MAX_SETPOINT_STEP}m"
        )

    def vehicle_status_callback(self, msg):
        """Vehicle durumunu takip et"""
        self.vehicle_status = msg
        self.arming_state = msg.arming_state
        self.nav_state = msg.nav_state

    def odom_callback(self, msg):
        self.current_pos_enu = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def route_wp_callback(self, msg: PoseStamped):
        """Route agent'ın ürettiği tek waypoint'i al."""
        self.route_wp = msg.pose
        self.route_wp_stamp = self.get_clock().now().nanoseconds * 1e-9

    def path_callback(self, msg):
        poses = msg.poses
        if not poses:
            self.current_path = []
            self.path_index = 0
            self.last_plan_start = None
            self.last_plan_end = None
            return

        start = poses[0].pose.position
        end = poses[-1].pose.position

        if self.last_plan_start is None or self.last_plan_end is None:
            self.current_path = poses
            self.path_index = 0
            self.last_plan_start = start
            self.last_plan_end = end
            return

        ds = math.sqrt((start.x - self.last_plan_start.x) ** 2 + (start.y - self.last_plan_start.y) ** 2)
        de = math.sqrt((end.x - self.last_plan_end.x) ** 2 + (end.y - self.last_plan_end.y) ** 2)
        plan_changed_a_lot = (ds > PATH_RESET_DIST) or (de > PATH_RESET_DIST)

        self.current_path = poses
        self.last_plan_start = start
        self.last_plan_end = end

        if plan_changed_a_lot:
            self.path_index = 0

    def timer_callback(self):
        """
        GELİŞTİRİLMİŞ: Sıralı ve kontrollü başlatma
        1. Önce yeterli setpoint gönder (en az 2 saniye = 40 döngü @ 50ms)
        2. Sonra offboard mode'a geç
        3. Mode geçişini kontrol et
        4. Son olarak arm et
        """
        # Her durumda setpoint ve control mode gönder
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

        # Başlatma sekansı
        if not self.initialization_complete:
            # Adım 1: En az 50 setpoint gönder (2.5 saniye @ 50ms)
            if self.offboard_setpoint_counter < 50:
                self.offboard_setpoint_counter += 1
                if self.offboard_setpoint_counter == 49:
                    self.get_logger().info("✓ Yeterli setpoint gönderildi, offboard mode'a geçiliyor...")
                return

            # Adım 2: Offboard mode'a geç (bir kere dene)
            if not self.offboard_mode_attempted:
                self.get_logger().info("→ Offboard mode komutu gönderiliyor...")
                self.publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
                )
                self.offboard_mode_attempted = True
                self.offboard_setpoint_counter += 1
                return

            # Adım 3: Offboard mode'un aktif olmasını bekle (en az 1 saniye = 20 döngü)
            if self.offboard_setpoint_counter < 70:  # 50 + 20
                self.offboard_setpoint_counter += 1
                
                # Nav state kontrolü: 14 = Offboard mode
                if self.vehicle_status and self.nav_state == 14:
                    self.get_logger().info("✓ Offboard mode aktif!")
                    self.offboard_setpoint_counter = 70  # Hızlıca arm adımına geç
                elif self.offboard_setpoint_counter == 69:
                    if self.vehicle_status and self.nav_state != 14:
                        self.get_logger().warn(
                            f"⚠ Offboard mode henüz aktif değil (nav_state: {self.nav_state}). "
                            "Yine de arm deneniyor..."
                        )
                return

            # Adım 4: Arm et (bir kere dene)
            if not self.arm_attempted:
                self.get_logger().info("→ Arm komutu gönderiliyor...")
                self.arm()
                self.arm_attempted = True
                self.offboard_setpoint_counter += 1
                return

            # Adım 5: Arm durumunu kontrol et (1 saniye bekle)
            if self.offboard_setpoint_counter < 90:  # 70 + 20
                self.offboard_setpoint_counter += 1
                
                # Arming state: 2 = Armed
                if self.vehicle_status and self.arming_state == 2:
                    self.get_logger().info("✓✓✓ Drone armed ve hazır! ✓✓✓")
                    self.initialization_complete = True
                elif self.offboard_setpoint_counter == 89:
                    if self.vehicle_status and self.arming_state != 2:
                        self.get_logger().error(
                            f"✗ Arm başarısız! (arming_state: {self.arming_state}). "
                            "QGroundControl'den manuel arm etmeyi deneyin."
                        )
                    else:
                        self.get_logger().warn("⚠ Vehicle status alınamıyor, arm durumu belirsiz.")
                    self.initialization_complete = True  # Yine de devam et
                return

        # Normal operasyon
        self.offboard_setpoint_counter += 1

    def _find_closest_index_nearby(self):
        if not self.current_path:
            return 0

        curr_x = self.current_pos_enu[0]
        curr_y = self.current_pos_enu[1]

        n = len(self.current_path)
        i0 = max(0, self.path_index - CLOSEST_SEARCH_WINDOW)
        i1 = min(n - 1, self.path_index + CLOSEST_SEARCH_WINDOW)

        best_i = self.path_index
        best_d2 = float('inf')

        for i in range(i0, i1 + 1):
            p = self.current_path[i].pose.position
            dx = p.x - curr_x
            dy = p.y - curr_y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        return best_i

    def _advance_path_index_if_reached(self):
        if not self.current_path:
            self.path_index = 0
            return

        self.path_index = self._find_closest_index_nearby()

        curr_x = self.current_pos_enu[0]
        curr_y = self.current_pos_enu[1]

        while self.path_index < len(self.current_path) - 1:
            wp = self.current_path[self.path_index].pose.position
            dist = math.sqrt((wp.x - curr_x) ** 2 + (wp.y - curr_y) ** 2)
            if dist < WAYPOINT_REACHED_DIST:
                self.path_index += 1
            else:
                break

    def _get_lookahead_along_path(self, lookahead_dist):
        if not self.current_path:
            return None

        n = len(self.current_path)
        if n == 1:
            return self.current_path[0].pose.position

        i = max(0, min(self.path_index, n - 1))

        acc = 0.0
        prev = self.current_path[i].pose.position

        for j in range(i + 1, n):
            cur = self.current_path[j].pose.position
            seg = math.sqrt((cur.x - prev.x) ** 2 + (cur.y - prev.y) ** 2)
            acc += seg
            if acc >= lookahead_dist:
                return cur
            prev = cur

        return self.current_path[-1].pose.position

    def get_adaptive_lookahead_point(self):
        if not self.current_path:
            return None, 0.0

        self._advance_path_index_if_reached()

        curr_x = self.current_pos_enu[0]
        curr_y = self.current_pos_enu[1]

        target_far = self._get_lookahead_along_path(MAX_LOOKAHEAD)
        if target_far is None:
            return None, 0.0

        dx = target_far.x - curr_x
        dy = target_far.y - curr_y
        desired_yaw = math.atan2(dy, dx)

        yaw_error = abs(
            math.atan2(
                math.sin(desired_yaw - self.current_yaw),
                math.cos(desired_yaw - self.current_yaw)
            )
        )

        if self.in_turn_mode:
            if yaw_error < TURN_OFF_THRESHOLD:
                self.in_turn_mode = False
        else:
            if yaw_error > TURN_ON_THRESHOLD:
                self.in_turn_mode = True

        if self.in_turn_mode:
            target_close = self._get_lookahead_along_path(MIN_LOOKAHEAD)
            final_target = target_close if target_close else target_far
            return final_target, yaw_error
        else:
            return target_far, yaw_error

    def _has_active_route_wp(self) -> bool:
        if self.route_wp is None:
            return False
        now = self.get_clock().now().nanoseconds * 1e-9
        return (now - self.route_wp_stamp) < ROUTE_WP_TIMEOUT

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()

        if self._has_active_route_wp():
            target_enu = self.route_wp.position
            q = self.route_wp.orientation
            wp_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

            t_north = target_enu.y
            t_east = target_enu.x
            t_down = -target_enu.z if target_enu.z > 0.0 else self.mission_altitude

            self.sp_n = t_north
            self.sp_e = t_east

            msg.position = [t_north, t_east, t_down]
            msg.velocity = [float('nan'), float('nan'), float('nan')]
            msg.yaw = math.atan2(
                math.sin(wp_yaw), math.cos(wp_yaw)
            )
            self.last_valid_yaw = msg.yaw

            msg.timestamp = self.get_clock().now().nanoseconds // 1000
            self.trajectory_setpoint_pub.publish(msg)
            return

        target_enu, yaw_error = self.get_adaptive_lookahead_point()

        if target_enu is not None:
            target_north = target_enu.y
            target_east = target_enu.x
            target_down = self.mission_altitude

            curr_north = self.current_pos_enu[1]
            curr_east = self.current_pos_enu[0]

            delta_north = target_north - curr_north
            delta_east = target_east - curr_east
            dist_to_target = math.sqrt(delta_north ** 2 + delta_east ** 2)

            raw_n = target_north
            raw_e = target_east

            if self.sp_n is None:
                self.sp_n, self.sp_e = raw_n, raw_e
            else:
                self.sp_n = (1.0 - SETPOINT_ALPHA) * self.sp_n + SETPOINT_ALPHA * raw_n
                self.sp_e = (1.0 - SETPOINT_ALPHA) * self.sp_e + SETPOINT_ALPHA * raw_e

                dn = self.sp_n - curr_north
                de = self.sp_e - curr_east
                d = math.sqrt(dn * dn + de * de)
                if d > MAX_SETPOINT_STEP and d > 1e-6:
                    scale = MAX_SETPOINT_STEP / d
                    self.sp_n = curr_north + dn * scale
                    self.sp_e = curr_east + de * scale

            msg.position = [self.sp_n, self.sp_e, target_down]
            msg.velocity = [float('nan'), float('nan'), float('nan')]

            if dist_to_target > NEAR_TARGET_DIST:
                desired_yaw_ned = math.atan2(delta_east, delta_north)

                yaw_err = math.atan2(
                    math.sin(desired_yaw_ned - self.last_valid_yaw),
                    math.cos(desired_yaw_ned - self.last_valid_yaw)
                )

                if abs(yaw_err) < YAW_ERR_DEADBAND and (yaw_error <= TURN_THRESHOLD):
                    msg.yaw = self.last_valid_yaw
                    if hasattr(msg, "yaw_speed"):
                        msg.yaw_speed = 0.0
                else:
                    msg.yaw = desired_yaw_ned
                    self.last_valid_yaw = desired_yaw_ned

                    if hasattr(msg, "yaw_speed"):
                        req_rate = yaw_err / max(TIMER_PERIOD, 1e-3)
                        if req_rate > MAX_YAW_RATE:
                            req_rate = MAX_YAW_RATE
                        if req_rate < -MAX_YAW_RATE:
                            req_rate = -MAX_YAW_RATE
                        msg.yaw_speed = req_rate
            else:
                msg.yaw = self.last_valid_yaw
                if hasattr(msg, "yaw_speed"):
                    msg.yaw_speed = 0.0

            if self.offboard_setpoint_counter % 20 == 0 and self.initialization_complete:
                mode_str = "SHORT (Viraj)" if self.in_turn_mode else "LONG (Düz)"
                self.get_logger().info(
                    f"Mode: {mode_str} | Dist: {dist_to_target:.2f}m | "
                    f"YawErrFar: {math.degrees(yaw_error):.1f}° | idx={self.path_index}"
                )

        else:
            msg.position = [self.current_pos_enu[1], self.current_pos_enu[0], self.mission_altitude]
            msg.velocity = [float('nan'), float('nan'), float('nan')]
            msg.yaw = float('nan')
            if hasattr(msg, "yaw_speed"):
                msg.yaw_speed = float('nan')

        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.trajectory_setpoint_pub.publish(msg)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.offboard_control_mode_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.vehicle_command_pub.publish(msg)

    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0
        )
        self.get_logger().info("Arm komutu gönderildi")


def main(args=None):
    rclpy.init(args=args)
    node = OffboardControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()