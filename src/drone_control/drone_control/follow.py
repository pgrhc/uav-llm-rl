#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math

# PX4 Mesajları
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand

# ROS Standart Mesajları
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry

class OffboardControl(Node):

    def __init__(self):
        super().__init__('offboard_control')

        # --- 1. QoS AYARLARI (KRİTİK BÖLÜM) ---
        # PX4 (UDP) için Best Effort şarttır.
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Nav2 ve Odometry için "Her şeyi kabul et" profili.
        # Bu ayar ile yayıncı Reliable da olsa Best Effort da olsa veriyi alırız.
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
        # Odometriyi dinle (Best Effort yaparak garantiye alıyoruz)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered', 
            self.odom_callback,
            qos_standard # <-- Değişti
        )
        
        # Path dinle (Nav2 topic ismi genelde /plan olabilir, /path ise burası doğru)
        self.path_sub = self.create_subscription(
            Path,
            '/plan', 
            self.path_callback,
            qos_standard # <-- Değişti
        )

        # Değişkenler
        self.offboard_setpoint_counter = 0
        self.current_path = []
        self.current_wp_index = 0
        self.current_pos_enu = [0.0, 0.0, 0.0] 
        self.mission_altitude = -5.0 
        self.acceptance_radius = 0.5
        
        # Debug için sayaç
        self.log_counter = 0

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("Offboard Node Başlatıldı. Path bekleniyor...")

    def odom_callback(self, msg):
        self.current_pos_enu = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]

    def path_callback(self, msg):
        self.current_path = msg.poses
        self.current_wp_index = 0
        self.get_logger().info(f"!!! YENİ ROTA ALINDI !!! Uzunluk: {len(self.current_path)} nokta.")

    def timer_callback(self):
        # 1 saniyede bir log bas (10 * 0.1s)
        self.log_counter += 1
        
        # Arm ve Mod geçişi
        if self.offboard_setpoint_counter == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
            self.arm()

        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        
        # Path var mı kontrolü
        if self.current_path and self.current_wp_index < len(self.current_path):
            
            target_pose_enu = self.current_path[self.current_wp_index].pose.position
            
            # Koordinat Dönüşümü (ENU -> NED)
            px4_north = target_pose_enu.y
            px4_east  = target_pose_enu.x
            px4_down  = self.mission_altitude

            # Mesafe Kontrolü
            dx = target_pose_enu.x - self.current_pos_enu[0]
            dy = target_pose_enu.y - self.current_pos_enu[1]
            dist_2d = math.sqrt(dx*dx + dy*dy)

            msg.position = [px4_north, px4_east, px4_down]
            
            # Yaw Hesabı (Basit: Hep 0/Kuzey baksın, hataları önlemek için)
            msg.yaw = 0.0 

            # Loglama (Hata ayıklama için çok önemli)
            if self.log_counter % 20 == 0: # 2 saniyede bir yaz
                self.get_logger().info(
                    f"Gidiliyor -> WP:{self.current_wp_index}/{len(self.current_path)} "
                    f"Dist:{dist_2d:.2f}m "
                    f"Hedef(NED):[{px4_north:.2f}, {px4_east:.2f}]"
                )

            if dist_2d < self.acceptance_radius:
                self.current_wp_index += 1
                self.get_logger().info(f"*** Waypoint {self.current_wp_index} Ulaşıldı! ***")

        else:
            # Path Yoksa: Havada Bekle (Hold)
            # Olduğumuz yerde (NED cinsinden) kalıyoruz
            # ENU y -> NED x (North), ENU x -> NED y (East)
            hold_north = self.current_pos_enu[1]
            hold_east  = self.current_pos_enu[0]
            
            msg.position = [hold_north, hold_east, self.mission_altitude]
            msg.yaw = float('nan')
            
            if self.log_counter % 30 == 0:
                self.get_logger().info("Path bekleniyor... (Havada sabit)")

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
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("Arm komutu gönderildi")

def main(args=None):
    rclpy.init(args=args)
    node = OffboardControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()