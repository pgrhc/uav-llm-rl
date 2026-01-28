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
from geometry_msgs.msg import PoseStamped # Hedef ve Çıkış için

class OffboardControl(Node):

    def __init__(self):
        super().__init__('offboard_control')

        # --- 1. QoS AYARLARI ---
        
        # PX4 (UDP) için Best Effort
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Nav2 ve Odom için (Volatile)
        qos_standard = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # LABİRENT ÇIKIŞI İÇİN ÖZEL QoS (Transient Local)
        # Spawner kodu mesajı biz gelmeden önce attığı için bu ayar ŞART.
        qos_transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- Publisherlar ---
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_px4)
        
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_px4)
        
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_px4)

        # Nav2'ye hedef göndermek için Publisher (/goal_pose)
        self.nav2_goal_pub = self.create_publisher(
            PoseStamped, '/goal_pose', qos_transient)

        # --- Subscriberlar ---
        
        # 1. Odometri
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, qos_standard)
        
        # 2. Nav2'nin oluşturduğu yol (/plan)
        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_callback, qos_standard)

        # 3. Labirent Çıkış Noktası (/maze/exit)
        self.exit_sub = self.create_subscription(
            PoseStamped, '/maze/exit', self.exit_callback, qos_transient)

        # Değişkenler
        self.offboard_setpoint_counter = 0
        self.current_path = []
        self.current_wp_index = 0
        self.current_pos_enu = [0.0, 0.0, 0.0] 
        self.mission_altitude = -5.0 
        self.acceptance_radius = 0.5
        self.log_counter = 0
        self.goal_sent_to_nav2 = False # Hedefi tekrar tekrar göndermemek için bayrak

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("🚀 Hazır! Labirent çıkış koordinatı bekleniyor...")

    # --- CALLBACK FONKSİYONLARI ---

    def exit_callback(self, msg):
        """
        Labirent oluşturucu kodundan (/maze/exit) çıkış koordinatını alır
        ve bunu Nav2'ye (/goal_pose) iletir.
        """
        if not self.goal_sent_to_nav2:
            self.get_logger().info(f"🎯 Labirent Çıkışı Alındı: X={msg.pose.position.x:.2f}, Y={msg.pose.position.y:.2f}")
            self.get_logger().info("📡 Nav2'ye hedef iletiliyor...")
            
            # Nav2'ye hedefi paslıyoruz
            self.nav2_goal_pub.publish(msg)
            self.goal_sent_to_nav2 = True
        else:
            # Zaten aldıysak tekrar işleme (spam engelleme)
            pass

    def odom_callback(self, msg):
        self.current_pos_enu = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]

    def path_callback(self, msg):
        # Nav2 yeni bir yol hesapladığında burası çalışır
        self.current_path = msg.poses
        self.current_wp_index = 0
        self.get_logger().info(f"✅ Nav2 Rota Oluşturdu! Uzunluk: {len(self.current_path)} nokta.")

    def timer_callback(self):
        self.log_counter += 1
        
        # Arm ve Offboard Moduna Geçiş (İlk 1 saniye içinde)
        if self.offboard_setpoint_counter == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
            self.arm()

        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        
        # Eğer elimizde takip edilecek bir rota varsa
        if self.current_path and self.current_wp_index < len(self.current_path):
            
            target_pose_enu = self.current_path[self.current_wp_index].pose.position
            
            # ENU -> NED Dönüşümü
            px4_north = target_pose_enu.y
            px4_east  = target_pose_enu.x
            px4_down  = self.mission_altitude

            # Mesafe Kontrolü
            dx = target_pose_enu.x - self.current_pos_enu[0]
            dy = target_pose_enu.y - self.current_pos_enu[1]
            dist_2d = math.sqrt(dx*dx + dy*dy)

            msg.position = [px4_north, px4_east, px4_down]
            msg.yaw = 0.0 # Basitlik için kuzeye bak

            if self.log_counter % 20 == 0:
                self.get_logger().info(
                    f"Gidiliyor -> WP:{self.current_wp_index}/{len(self.current_path)} "
                    f"Dist:{dist_2d:.2f}m "
                )

            # Waypoint'e yaklaştıysak bir sonrakine geç
            if dist_2d < self.acceptance_radius:
                self.current_wp_index += 1
                
                # Eğer son noktaya geldiysek
                if self.current_wp_index >= len(self.current_path):
                     self.get_logger().info("🏆 LABİRENT ÇIKIŞINA ULAŞILDI! (Hover Modu)")

        else:
            # Path Yoksa: Havada Bekle (Hold)
            # Eğer hedefi Nav2'ye gönderdik ama henüz path gelmediyse burada bekler
            hold_north = self.current_pos_enu[1]
            hold_east  = self.current_pos_enu[0]
            
            msg.position = [hold_north, hold_east, self.mission_altitude]
            msg.yaw = float('nan')
            
            if self.log_counter % 30 == 0:
                if self.goal_sent_to_nav2:
                    self.get_logger().info("⏳ Nav2 rota hesaplıyor, bekleniyor...")
                else:
                    self.get_logger().info("🛑 Çıkış koordinatı bekleniyor... (Henüz /maze/exit gelmedi)")

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