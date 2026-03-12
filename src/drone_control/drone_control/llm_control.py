#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
import json
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# ✅ PX4 mesajları
from px4_msgs.msg import (
    TrajectorySetpoint,
    OffboardControlMode,
    VehicleCommand,
    VehicleControlMode
)

class DroneController(Node):
    def __init__(self):
        super().__init__('drone_controller')
        
        # State
        self.current_mode = "normal"
        self.speed_limit = 1.5
        self.target_velocity = Twist()
        self.offboard_counter = 0
        self.is_armed = False

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ✅ PX4 Publishers
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            10
        )
        
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            10
        )
        
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            10
        )
        
        # ✅ LLM Subscribers
        self.create_subscription(
            String, '/llm/flight_mode', 
            self.mode_callback, 10
        )
        
        self.create_subscription(
            Twist, '/llm/velocity_command', 
            self.velocity_callback, 10
        )
        
        self.create_subscription(
            Float32, '/llm/speed_limit', 
            self.speed_limit_callback, 10
        )
        
        self.create_subscription(
            String, '/llm/strategic_decision', 
            self.decision_callback, 10
        )
        
        # ✅ PX4 Status Subscriber
        self.create_subscription(
            VehicleControlMode,
            '/fmu/out/vehicle_control_mode',
            self.control_mode_callback,
            10
        )
        
        # ✅ Control loop - 20Hz (PX4 için önemli!)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("🚁 PX4 Drone Controller başlatıldı")
        self.get_logger().info("⚠️  ARM etmek için komut gönderin...")
    
    def control_mode_callback(self, msg):
        """PX4 control mode durumu"""
        if msg.flag_armed and not self.is_armed:
            self.get_logger().info("✅ Drone ARM edildi!")
            self.is_armed = True
        elif not msg.flag_armed and self.is_armed:
            self.get_logger().warn("⚠️ Drone DISARM edildi!")
            self.is_armed = False
    
    def mode_callback(self, msg):
        """LLM mode değişikliği"""
        old_mode = self.current_mode
        self.current_mode = msg.data
        
        if old_mode != self.current_mode:
            self.get_logger().info(f"🔄 Mode: {old_mode} → {self.current_mode}")
            
            if self.current_mode == "emergency_stop":
                self.emergency_stop()
    
    def velocity_callback(self, msg):
        """LLM'den velocity komutu"""
        self.target_velocity = msg
        self.get_logger().info(
            f"📍 LLM Velocity: x={msg.linear.x:.2f}, "
            f"y={msg.linear.y:.2f}, z={msg.linear.z:.2f}"
        )
    
    def speed_limit_callback(self, msg):
        """Speed limit güncelleme"""
        self.speed_limit = msg.data
        self.get_logger().info(f"⚡ Speed limit: {msg.data:.2f} m/s")
    
    def decision_callback(self, msg):
        """Tam LLM kararı"""
        try:
            decision = json.loads(msg.data)
            action = decision.get("action", "maintain")
            self.get_logger().info(f"🧠 LLM Action: {action}")
        except:
            pass
    
    def control_loop(self):
        """Ana kontrol döngüsü - 20Hz"""
        
        # 1. Offboard control mode heartbeat (sürekli gönder!)
        self.publish_offboard_control_mode()
        
        # 2. İlk 10 döngü: ARM ve OFFBOARD mode aktif et
        if self.offboard_counter < 10:
            self.offboard_counter += 1
            if self.offboard_counter == 10:
                self.get_logger().info("🔓 ARM ve OFFBOARD mode aktifleştiriliyor...")
                self.arm()
                self.set_offboard_mode()
        
        # 3. Velocity setpoint gönder
        limited_vel = self.apply_speed_limit(self.target_velocity)
        self.publish_trajectory_setpoint(limited_vel)
    
    def publish_offboard_control_mode(self):
        """Offboard control mode - SÜREKLI GÖNDERİLMELİ!"""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True  # ✅ Velocity control
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        self.offboard_mode_pub.publish(msg)
    
    def publish_trajectory_setpoint(self, vel):
        """Twist → PX4 TrajectorySetpoint"""
        msg = TrajectorySetpoint()
        
        # ✅ Velocity (NED frame)
        msg.velocity[0] = float(vel.linear.x)   # North (forward)
        msg.velocity[1] = float(vel.linear.y)   # East (right)  
        msg.velocity[2] = float(-vel.linear.z)  # Down (negative up!)
        
        # Yaw rate
        msg.yawspeed = float(vel.angular.z)
        
        # Timestamp
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        self.trajectory_pub.publish(msg)
        
        # Debug log (her 20 frame'de bir - her saniye)
        if self.offboard_counter % 20 == 0:
            self.get_logger().info(
                f"➡️  Trajectory: vx={msg.velocity[0]:.2f}, "
                f"vy={msg.velocity[1]:.2f}, vz={msg.velocity[2]:.2f}"
            )
    
    def arm(self):
        """Drone'u ARM et"""
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0  # ARM
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        self.vehicle_command_pub.publish(msg)
        self.get_logger().info("🔓 ARM komutu gönderildi")
    
    def set_offboard_mode(self):
        """OFFBOARD mode'a geç"""
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0  # Custom mode
        msg.param2 = 6.0  # OFFBOARD mode
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        self.vehicle_command_pub.publish(msg)
        self.get_logger().info("🎮 OFFBOARD mode komutu gönderildi")
    
    def emergency_stop(self):
        """Emergency stop"""
        stop_vel = Twist()
        self.target_velocity = stop_vel
        self.get_logger().warn("🛑 EMERGENCY STOP!")
    
    def apply_speed_limit(self, vel):
        """Speed limit uygula"""
        limited = Twist()
        
        horizontal = math.sqrt(vel.linear.x**2 + vel.linear.y**2)
        if horizontal > self.speed_limit:
            scale = self.speed_limit / horizontal
            limited.linear.x = vel.linear.x * scale
            limited.linear.y = vel.linear.y * scale
        else:
            limited.linear.x = vel.linear.x
            limited.linear.y = vel.linear.y
        
        max_vertical = self.speed_limit * 0.6
        limited.linear.z = max(-max_vertical, min(max_vertical, vel.linear.z))
        limited.angular.z = vel.angular.z
        
        return limited

def main(args=None):
    rclpy.init(args=args)
    controller = DroneController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()