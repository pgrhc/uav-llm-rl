#!/usr/bin/env python3
"""
LLM Executor Node - ROS2 / PX4

Desteklenen action'lar (/llm/parsed_command topic'ine JSON):
  arm            -> drone'u arm et
  disarm         -> drone'u disarm et
  takeoff        -> varsayılan yükseklikte kalk (default 2.5m)
  takeoff_x      -> delta_z kadar kalk
  land           -> bulunduğu yerde in (PX4 NAV_LAND)
  return_home    -> kalkış noktasına dön, otomatik land
  hover          -> mevcut konumda kal
  ascend         -> yukarı
  descend        -> aşağı
  move_forward   -> ileri
  move_backward  -> geri
  move_right     -> sağa
  move_left      -> sola
  rotate_cw      -> saat yönünde
  rotate_ccw     -> saat tersine
  composite      -> çok eksenli
"""

import json
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from std_msgs.msg import String
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleOdometry,
    VehicleStatus,
)

MAX_ALTITUDE_NED  = -0.5    
MIN_ALTITUDE_NED  = -50.0 
MAX_HORIZ_STEP    = 30.0    
ARRIVAL_THRESHOLD = 0.5    
DEFAULT_TAKEOFF_H = 2.5  


class LLMExecutorNode(Node):

    def __init__(self):
        super().__init__("llm_executor_node")
        self.declare_parameter("px4_namespace",     "")
        self.declare_parameter("timer_period",      0.1)
        self.declare_parameter("arrival_threshold", ARRIVAL_THRESHOLD)
        self.declare_parameter("default_takeoff_h", DEFAULT_TAKEOFF_H)
        self.declare_parameter("max_altitude_ned",  MIN_ALTITUDE_NED)
        self.declare_parameter("min_altitude_ned",  MAX_ALTITUDE_NED)

        ns = self.get_parameter("px4_namespace").value.strip()
        self.timer_period      = float(self.get_parameter("timer_period").value)
        self.arrival_threshold = float(self.get_parameter("arrival_threshold").value)
        self.default_takeoff_h = float(self.get_parameter("default_takeoff_h").value)
        self.max_alt_ned       = float(self.get_parameter("max_altitude_ned").value)
        self.min_alt_ned       = float(self.get_parameter("min_altitude_ned").value)

        self.topic_prefix = f"/{ns}" if ns else ""
        px4_sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        px4_pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pub_offboard = self.create_publisher(
            OffboardControlMode,
            f"{self.topic_prefix}/fmu/in/offboard_control_mode",
            px4_pub_qos,
        )
        self.pub_traj = self.create_publisher(
            TrajectorySetpoint,
            f"{self.topic_prefix}/fmu/in/trajectory_setpoint",
            px4_pub_qos,
        )
        self.pub_cmd = self.create_publisher(
            VehicleCommand,
            f"{self.topic_prefix}/fmu/in/vehicle_command",
            px4_pub_qos,
        )
        self.create_subscription(
            VehicleOdometry,
            f"{self.topic_prefix}/fmu/out/vehicle_odometry",
            self._cb_odom, px4_sub_qos,
        )
        self.create_subscription(
            VehicleStatus,
            f"{self.topic_prefix}/fmu/out/vehicle_status_v1",
            self._cb_status, px4_sub_qos,
        )
        self.create_subscription(
            String, "/llm/parsed_command",
            self._cb_command, 10,
        )

        self.current_x:   Optional[float] = None
        self.current_y:   Optional[float] = None
        self.current_z:   Optional[float] = None
        self.current_yaw: float = 0.0

        self.target_x:   Optional[float] = None
        self.target_y:   Optional[float] = None
        self.target_z:   Optional[float] = None
        self.target_yaw: float = 0.0
        self.home_x: Optional[float] = None
        self.home_y: Optional[float] = None
        self.home_z: Optional[float] = None

        self.pending_cmd = None
        self.active_cmd  = None

        self.in_offboard_mode = False
        self.is_armed         = False
        self.engage_counter   = 0
        self._has_jerk = hasattr(TrajectorySetpoint(), "jerk")

        self._log_tick = 0

        self.timer = self.create_timer(self.timer_period, self._cb_timer)
        self.get_logger().info(
            f"LLMExecutorNode hazır | prefix='{self.topic_prefix or '/'}' | "
            "Bağlantı bekleniyor, otomatik arm YOK."
        )

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    @staticmethod
    def _quat_to_yaw(q) -> float:
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    @staticmethod
    def _norm(a: float) -> float:
        while a >  math.pi: a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    def _clamp_z(self, z: float) -> float:
        return max(self.max_alt_ned, min(self.min_alt_ned, z))

    def _clamp_horiz(self, cur: float, tgt: float) -> float:
        d = tgt - cur
        if abs(d) > MAX_HORIZ_STEP:
            d = math.copysign(MAX_HORIZ_STEP, d)
        return cur + d

    def _has_arrived(self) -> bool:
        if any(v is None for v in [
            self.current_x, self.current_y, self.current_z,
            self.target_x,  self.target_y,  self.target_z,
        ]):
            return False
        dist = math.sqrt(
            (self.current_x - self.target_x)**2 +
            (self.current_y - self.target_y)**2 +
            (self.current_z - self.target_z)**2
        )
        self._log_tick += 1
        if self._log_tick % 10 == 0:
            self.get_logger().info(
                f"[Mesafe] {dist:.2f}m | "
                f"cur_z={self.current_z:.2f} → tgt_z={self.target_z:.2f}"
            )
        return dist < self.arrival_threshold

    def _pub_vehicle_cmd(self, command: int, **kw):
        msg = VehicleCommand()
        msg.timestamp        = self._now_us()
        msg.command          = command
        msg.param1           = float(kw.get("param1", 0.0))
        msg.param2           = float(kw.get("param2", 0.0))
        msg.param3           = float(kw.get("param3", 0.0))
        msg.param4           = float(kw.get("param4", 0.0))
        msg.param5           = float(kw.get("param5", 0.0))
        msg.param6           = float(kw.get("param6", 0.0))
        msg.param7           = float(kw.get("param7", 0.0))
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        self.pub_cmd.publish(msg)

    def _send_arm(self):
        self._pub_vehicle_cmd(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("ARM komutu gönderildi.")

    def _send_disarm(self):
        self._pub_vehicle_cmd(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info("DISARM komutu gönderildi.")

    def _send_land(self):
        self._pub_vehicle_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("LAND komutu gönderildi.")

    def _engage_offboard(self):
        self._pub_vehicle_cmd(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

    def _pub_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp    = self._now_us()
        msg.position     = True
        msg.velocity     = False
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        self.pub_offboard.publish(msg)

    def _pub_setpoint(self):
        if any(v is None for v in [self.target_x, self.target_y, self.target_z]):
            return
        nan = float("nan")
        msg = TrajectorySetpoint()
        msg.timestamp    = self._now_us()
        msg.position     = [float(self.target_x),
                             float(self.target_y),
                             float(self.target_z)]
        msg.velocity     = [nan, nan, nan]
        msg.acceleration = [nan, nan, nan]
        msg.yaw          = float(self.target_yaw)
        msg.yawspeed     = nan
        if self._has_jerk:
            msg.jerk = [nan, nan, nan]
        self.pub_traj.publish(msg)

    def _cb_odom(self, msg: VehicleOdometry):
        self.current_x   = float(msg.position[0])
        self.current_y   = float(msg.position[1])
        self.current_z   = float(msg.position[2])
        self.current_yaw = self._quat_to_yaw(msg.q)
        if self.target_x is None:
            self.target_x   = self.current_x
            self.target_y   = self.current_y
            self.target_z   = self.current_z
            self.target_yaw = self.current_yaw
            self.get_logger().info(
                f"Bağlantı kuruldu: ({self.current_x:.2f}, "
                f"{self.current_y:.2f}, {self.current_z:.2f})"
            )

    def _cb_status(self, msg: VehicleStatus):
        was_offboard = self.in_offboard_mode
        self.in_offboard_mode = (
            msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        )
        if not was_offboard and self.in_offboard_mode:
            self.get_logger().info("PX4 offboard moda geçti.")
        elif was_offboard and not self.in_offboard_mode:
            self.get_logger().warning(
                f"Offboard moddan çıkıldı! nav_state={msg.nav_state}"
            )

        was_armed = self.is_armed
        self.is_armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        if not was_armed and self.is_armed:
            self.get_logger().info("Drone ARM edildi.")
            if self.current_x is not None and self.home_x is None:
                self.home_x = self.current_x
                self.home_y = self.current_y
                self.home_z = self.current_z
                self.get_logger().info(
                    f"Home noktası kaydedildi: "
                    f"({self.home_x:.2f}, {self.home_y:.2f}, {self.home_z:.2f})"
                )
        elif was_armed and not self.is_armed:
            self.get_logger().info("Drone DISARM edildi.")
            self.home_x = None 

    def _cb_command(self, msg: String):
        raw = msg.data.strip()
        self.get_logger().info(f"Komut alındı: {raw}")

        if self.current_x is None:
            self.get_logger().warning("Odom yok, komut reddedildi.")
            return

        try:
            cmd = json.loads(raw)
        except Exception as e:
            self.get_logger().error(f"JSON parse hatası: {e}")
            return

        self.pending_cmd = cmd

    def _handle_special(self, action: str, cmd: dict) -> bool:
        if action == "arm":
            if self.is_armed:
                self.get_logger().warning("Zaten arm edilmiş.")
            else:
                self._send_arm()
            return True

        if action == "disarm":
            if not self.is_armed:
                self.get_logger().warning("Zaten disarm durumunda.")
            else:
                self._send_disarm()
            return True
        if action in ("takeoff", "takeoff_x"):
            if not self.is_armed:
                self.get_logger().warning("Takeoff için önce arm et!")
                return True
            height = float(cmd.get("delta_z", self.default_takeoff_h))
            if height <= 0:
                height = self.default_takeoff_h
            target_z = self._clamp_z(self.current_z - height)
            self.target_x   = self.current_x
            self.target_y   = self.current_y
            self.target_z   = target_z
            self.target_yaw = self.current_yaw
            self.active_cmd = {"action": action}
            self._engage_offboard()
            self.get_logger().info(
                f"Takeoff: {height:.1f}m → target_z={target_z:.2f} (NED)"
            )
            return True
        if action == "land":
            self._send_land()
            self.active_cmd = None
            return True
        if action in ("return_home", "rth", "rtl"):
            if self.home_x is None:
                self.get_logger().warning("Home noktası bilinmiyor!")
                return True
            safe_z = self._clamp_z(self.home_z - self.default_takeoff_h)
            self.target_x   = self.home_x
            self.target_y   = self.home_y
            self.target_z   = safe_z
            self.target_yaw = self.current_yaw
            self.active_cmd = {"action": "return_home"}
            self.get_logger().info(
                f"RTH: ({self.home_x:.2f}, {self.home_y:.2f}, {safe_z:.2f})"
            )
            return True

        return False

    def _build_target(self, cmd: dict) -> dict:
        body_dx  = float(cmd.get("delta_x", 0.0))
        body_dy  = float(cmd.get("delta_y", 0.0))
        dz       = float(cmd.get("delta_z", 0.0))
        dyaw_deg = float(cmd.get("delta_yaw", 0.0))
        action   = cmd.get("action", "hover")

        yaw = self.current_yaw
        world_dx = body_dx * math.cos(yaw) - body_dy * math.sin(yaw)
        world_dy = body_dx * math.sin(yaw) + body_dy * math.cos(yaw)

        target_x   = self._clamp_horiz(self.current_x, self.current_x + world_dx)
        target_y   = self._clamp_horiz(self.current_y, self.current_y + world_dy)
        target_z   = self._clamp_z(self.current_z - dz)
        target_yaw = self._norm(self.current_yaw + math.radians(dyaw_deg))

        return {
            "action":     action,
            "target_x":   target_x,
            "target_y":   target_y,
            "target_z":   target_z,
            "target_yaw": target_yaw,
            "reasoning":  cmd.get("reasoning", ""),
        }

    def _cb_timer(self):
        if self.current_x is None:
            return
        self._pub_offboard_mode()
        self._pub_setpoint()
        if not self.in_offboard_mode:
            self.engage_counter += 1
            if self.engage_counter % 20 == 0:
                self._engage_offboard()
            return
        if self.pending_cmd is not None:
            action = self.pending_cmd.get("action", "hover")
            if self._handle_special(action, self.pending_cmd):
                self.pending_cmd = None
                return
            if action == "hover":
                self.target_x   = self.current_x
                self.target_y   = self.current_y
                self.target_z   = self.current_z
                self.target_yaw = self.current_yaw
                self.active_cmd = None
                self.pending_cmd = None
                self.get_logger().info("Hover: mevcut konumda tutuluyor.")
                return
            target = self._build_target(self.pending_cmd)
            self.target_x   = target["target_x"]
            self.target_y   = target["target_y"]
            self.target_z   = target["target_z"]
            self.target_yaw = target["target_yaw"]
            self.active_cmd = target
            self.pending_cmd = None
            self.get_logger().info(
                f"Yürütülüyor: {target['action']} → "
                f"({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}) "
                f"yaw={math.degrees(self.target_yaw):.1f}°"
            )


        if self.active_cmd is not None and self._has_arrived():
            action = self.active_cmd.get("action", "")
            self.get_logger().info(f"Hedefe ulaşıldı: {action}")
            if action == "return_home":
                self.get_logger().info("Home'a ulaşıldı → Land.")
                self._send_land()
            self.active_cmd = None

def main(args=None):
    rclpy.init(args=args)
    node = LLMExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Kapatılıyor...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()