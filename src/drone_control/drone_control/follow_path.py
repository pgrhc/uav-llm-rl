#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleStatus

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Path
from std_msgs.msg import Bool


MAX_LOOKAHEAD = 1.5
MIN_LOOKAHEAD = 0.6
TURN_THRESHOLD = 0.5
YAW_DEADBAND = 0.2

TURN_ON_THRESHOLD  = 0.55
TURN_OFF_THRESHOLD = 0.35

SETPOINT_ALPHA = 0.85
MAX_SETPOINT_STEP = 1.50
TIMER_PERIOD = 0.05
MAX_ALLOWED_DIST_FROM_CURRENT = 2.0

NEAR_TARGET_DIST = 0.15
WAYPOINT_REACHED_DIST = 0.35

YAW_ERR_DEADBAND = 0.08
MAX_YAW_RATE = 0.6

CLOSEST_SEARCH_WINDOW = 50
PATH_RESET_DIST = 1.0

ROUTE_WP_TIMEOUT = 0.8

SLOW_DOWN_DIST = 1.2  

MIN_ALTITUDE = 0.3
MAX_ALTITUDE = 3.0


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

        self.declare_parameter('max_speed', float(MAX_SETPOINT_STEP))
        self.declare_parameter('alpha', float(SETPOINT_ALPHA))
        self.declare_parameter('route_waypoint_topic', '/route/waypoint_desired')

        self.MAX_SETPOINT_STEP = float(self.get_parameter('max_speed').value)
        self.SETPOINT_ALPHA = float(self.get_parameter('alpha').value)

        route_wp_topic = str(self.get_parameter('route_waypoint_topic').value).strip()
        if route_wp_topic and not route_wp_topic.startswith('/'):
            route_wp_topic = '/' + route_wp_topic

        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_px4)
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_px4)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_px4)

        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, qos_standard)
        self.route_wp_sub = self.create_subscription(
            PoseStamped, route_wp_topic, self.route_wp_callback, 10)
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_px4)

        
        self.min_lidar_dist = 999.0
        self.lidar_sub = self.create_subscription(
            Float32MultiArray, '/threat/state_vec', self.threat_cb, 10)

        self.offboard_setpoint_counter = 0
        self.current_pos_enu = [0.0, 0.0, 0.0]
        self.current_yaw = 0.0
        self.mission_altitude = -1.2
        self.last_valid_yaw = 0.0

        self.sp_n = None
        self.sp_e = None

        self.route_wp = None
        self.route_wp_stamp = 0.0

        self.vehicle_status = None
        self.arming_state = 0
        self.nav_state = 0
        self.offboard_mode_attempted = False
        self.arm_attempted = False
        self.initialization_complete = False
        self.recovery_mode = False
        self.recovery_path = []
        self.recovery_idx = 0
        self.recovery_done_pub = self.create_publisher(
            Bool, "/route/recovery_done", 10
        )

        self.recovery_mode_sub = self.create_subscription(
            Bool, "/route/recovery_mode", self.recovery_mode_callback, 10
        )

        self.plan_sub = self.create_subscription(
            Path, "/route/recovery_plan", self.plan_callback, 10
        )
        self.timer = self.create_timer(TIMER_PERIOD, self.timer_callback)
        self.get_logger().info(
            f"ADAPTİF MOD: Timer={TIMER_PERIOD}s | Alpha={self.SETPOINT_ALPHA} | "
            f"MaxStep={self.MAX_SETPOINT_STEP}m | route_wp={route_wp_topic} | "
            f"EmergencyStop=KAPALI (kaçış env tarafından yönetilir)"
        )

    def threat_cb(self, msg):
        data = msg.data
        if len(data) >= 39:
            lidar = [data[i] for i in range(3, 39) if data[i] > 1e-3]
            if lidar:
                self.min_lidar_dist = min(lidar) * 30.0
            else:
                self.min_lidar_dist = 999.0

    def recovery_mode_callback(self, msg: Bool):
        self.recovery_mode = bool(msg.data)

        if self.recovery_mode:
            self.get_logger().warn("RECOVERY MODE aktif: RL waypoint yok sayılacak, /plan takip edilecek.")
            self.route_wp = None
            self.route_wp_stamp = 0.0
            self.sp_n = None
            self.sp_e = None
            self.recovery_idx = 0
        else:
            self.get_logger().info("RECOVERY MODE kapandı: RL waypoint tekrar aktif.")


    def plan_callback(self, msg: Path):
        if not self.recovery_mode:
            return

        self.recovery_path = [p.pose for p in msg.poses]
        self.recovery_idx = 0

        self.get_logger().warn(
            f"Recovery A* path alındı: {len(self.recovery_path)} waypoint"
        )

    def _get_recovery_target_pose(self):
        if not self.recovery_mode:
            return None

        if not self.recovery_path:
            return None

        curr_x = self.current_pos_enu[0]
        curr_y = self.current_pos_enu[1]

        while self.recovery_idx < len(self.recovery_path):
            pose = self.recovery_path[self.recovery_idx]
            px = pose.position.x
            py = pose.position.y

            dist = math.hypot(px - curr_x, py - curr_y)

            if dist < WAYPOINT_REACHED_DIST:
                self.recovery_idx += 1
            else:
                return pose

        if self.recovery_idx >= len(self.recovery_path):
            done = Bool()
            done.data = True
            self.recovery_done_pub.publish(done)
            self.recovery_mode = False

        self.get_logger().warn("Recovery path tamamlandı. Drone spawn/0,0 noktasına döndü.")

        self.recovery_mode = False
        self.recovery_path = []
        self.recovery_idx = 0
        self.sp_n = None
        self.sp_e = None

        return None
    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg
        self.arming_state = msg.arming_state
        self.nav_state = msg.nav_state

    def odom_callback(self, msg):
        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y
        new_z = msg.pose.pose.position.z

        if hasattr(self, '_first_odom_received') and self._first_odom_received:
            dist_jump = math.hypot(new_x - self.current_pos_enu[0], new_y - self.current_pos_enu[1])
            if dist_jump > 2.0:
                self.get_logger().warn(f"🚀 Teleport detected ({dist_jump:.1f}m jump)! Clearing controller state.")
                self.sp_n = None
                self.sp_e = None
                self.route_wp = None
                self.route_wp_stamp = 0.0

        self.current_pos_enu = [new_x, new_y, new_z]
        self._first_odom_received = True

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def route_wp_callback(self, msg: PoseStamped):
        self.route_wp = msg.pose
        self.route_wp_frame_id = msg.header.frame_id
        self.route_wp_stamp = self.get_clock().now().nanoseconds * 1e-9

    def timer_callback(self):
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

        if not self.initialization_complete:
            if self.offboard_setpoint_counter < 50:
                self.offboard_setpoint_counter += 1
                if self.offboard_setpoint_counter == 49:
                    self.get_logger().info("✓ Yeterli setpoint gönderildi, offboard mode'a geçiliyor...")
                return

            if not self.offboard_mode_attempted:
                self.get_logger().info("→ Offboard mode komutu gönderiliyor...")
                self.publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
                )
                self.offboard_mode_attempted = True
                self.offboard_setpoint_counter += 1
                return

            if self.offboard_setpoint_counter < 70:
                self.offboard_setpoint_counter += 1
                if self.vehicle_status and self.nav_state == 14:
                    self.get_logger().info("✓ Offboard mode aktif!")
                    self.offboard_setpoint_counter = 70
                elif self.offboard_setpoint_counter == 69:
                    if self.vehicle_status and self.nav_state != 14:
                        self.get_logger().warn(
                            f"⚠ Offboard mode henüz aktif değil (nav_state: {self.nav_state}). "
                            "Yine de arm deneniyor..."
                        )
                return

            if not self.arm_attempted:
                self.get_logger().info("→ Arm komutu gönderiliyor...")
                self.arm()
                self.arm_attempted = True
                self.offboard_setpoint_counter += 1
                return

            if self.offboard_setpoint_counter < 90:
                self.offboard_setpoint_counter += 1
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
                    self.initialization_complete = True
                return

        self.offboard_setpoint_counter += 1

    def _has_active_route_wp(self) -> bool:
        if self.route_wp is None:
            return False
        now = self.get_clock().now().nanoseconds * 1e-9
        return (now - self.route_wp_stamp) < ROUTE_WP_TIMEOUT

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()

        target_north = None
        target_east = None
        target_down = self.mission_altitude
        rl_yaw_bias_ned = 0.0

        # Mevcut pozisyon (Odom'dan gelen ENU verisini NED olarak kullanıyoruz)
        curr_north = self.current_pos_enu[1]
        curr_east = self.current_pos_enu[0]
        curr_down = -self.current_pos_enu[2]

        active_pose = None

        if self.recovery_mode:
            active_pose = self._get_recovery_target_pose()
        elif self._has_active_route_wp():
            active_pose = self.route_wp

        if active_pose is not None:
            t_enu = active_pose.position
            q = active_pose.orientation

            # ENU -> NED dönüşümü
            target_north = t_enu.y
            target_east = t_enu.x
            target_down = -t_enu.z if t_enu.z > 0.0 else self.mission_altitude

            # Altitude clamp (Güvenlik için)
            target_down = max(-MAX_ALTITUDE, min(-MIN_ALTITUDE, target_down))

            # --- Yaw Hesaplama (ENU to NED) ---
            # 1. Waypoint'in kendi içindeki yönelimi (Quat to Euler ENU)
            wp_yaw_enu = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            # 2. ENU Yaw -> NED Yaw dönüşümü
            wp_yaw_ned = -wp_yaw_enu + math.pi / 2.0
            
            # 3. Mevcut Yaw'ı da NED referansına çekelim (Odom callback'te ENU geliyordu)
            curr_yaw_ned = -self.current_yaw + math.pi / 2.0

            # 4. RL'den gelen bias (hata payını normalize et)
            yaw_bias_err = math.atan2(
                math.sin(wp_yaw_ned - curr_yaw_ned),
                math.cos(wp_yaw_ned - curr_yaw_ned)
            )
            rl_yaw_bias_ned = max(min(0.35 * yaw_bias_err, 0.25), -0.25)

        if target_north is not None and target_east is not None:
            # Mesafe kısıtlama (Çok uzaksa kırp)
            dist_to_wp_raw = math.hypot(target_north - curr_north, target_east - curr_east)
            
            if dist_to_wp_raw > MAX_ALLOWED_DIST_FROM_CURRENT:
                ratio = MAX_ALLOWED_DIST_FROM_CURRENT / dist_to_wp_raw
                target_north = curr_north + (target_north - curr_north) * ratio
                target_east = curr_east + (target_east - curr_east) * ratio

            # Yumuşatma (Alpha Filter)
            if self.sp_n is None:
                self.sp_n, self.sp_e = target_north, target_east
            else:
                alpha = self.SETPOINT_ALPHA
                max_step = self.MAX_SETPOINT_STEP

                self.sp_n = (1.0 - alpha) * self.sp_n + alpha * target_north
                self.sp_e = (1.0 - alpha) * self.sp_e + alpha * target_east

                # Adım boyu kısıtlama
                dn = self.sp_n - curr_north
                de = self.sp_e - curr_east
                d = math.hypot(dn, de)
                if d > max_step and d > 1e-6:
                    scale = max_step / d
                    self.sp_n = curr_north + dn * scale
                    self.sp_e = curr_east + de * scale

            # --- Lidar Soft Slowdown ---
            if self.min_lidar_dist < SLOW_DOWN_DIST:
                slow_factor = max(0.15, self.min_lidar_dist / SLOW_DOWN_DIST)
                self.sp_n = curr_north + (self.sp_n - curr_north) * slow_factor
                self.sp_e = curr_east + (self.sp_e - curr_east) * slow_factor

            msg.position = [float(self.sp_n), float(self.sp_e), float(target_down)]
            
            # --- Final Yaw Kontrolü ---
            # Hedef yönüne bakma (Look-ahead yaw)
            delta_n = target_north - curr_north
            delta_e = target_east - curr_east
            dist_to_target = math.hypot(delta_n, delta_e)

            if dist_to_target > NEAR_TARGET_DIST:
                base_yaw_ned = math.atan2(delta_e, delta_n)
            else:
                base_yaw_ned = self.last_valid_yaw

            desired_yaw_ned = base_yaw_ned + rl_yaw_bias_ned
            
            # Normalize yaw
            desired_yaw_ned = math.atan2(math.sin(desired_yaw_ned), math.cos(desired_yaw_ned))

            # Yaw hız kısıtlama
            yaw_err = math.atan2(math.sin(desired_yaw_ned - self.last_valid_yaw),
                                 math.cos(desired_yaw_ned - self.last_valid_yaw))
            
            yaw_step = max(min(0.5 * yaw_err, 0.15), -0.15)
            msg.yaw = self.last_valid_yaw + yaw_step
            self.last_valid_yaw = msg.yaw

        else:
            # Waypoint yoksa olduğu yerde dur
            msg.position = [curr_north, curr_east, float(self.mission_altitude)]
            msg.yaw = self.last_valid_yaw

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