#!/usr/bin/env python3
"""
Labirent Navigasyon Scripti
RTAB-Map + Nav2 kullanarak labirentten çıkış
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import sys
import os
import time
import subprocess
import math
import numpy as np

# Nav2 action ve mesaj tipleri
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleCommand

# Labirent fonksiyonlarını import et
# maze_deneme.py root dizinde (workspace root)
import importlib.util
# Workspace root'u bul (src/drone_control/drone_control/ -> ../../)
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
maze_deneme_path = os.path.join(workspace_root, 'maze_deneme.py')

if not os.path.exists(maze_deneme_path):
    raise FileNotFoundError(f"maze_deneme.py bulunamadı: {maze_deneme_path}")

spec = importlib.util.spec_from_file_location("maze_deneme", maze_deneme_path)
maze_deneme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze_deneme)

spawn_maze = maze_deneme.spawn_maze
ROWS = maze_deneme.ROWS
COLS = maze_deneme.COLS
CELL_SIZE = maze_deneme.CELL_SIZE


class MazeNavigator(Node):
    def __init__(self):
        super().__init__('maze_navigator')
        
        # Labirent parametreleri
        self.ROWS = ROWS
        self.COLS = COLS
        self.CELL_SIZE = CELL_SIZE
        
        # Labirent boyutu hesapla
        self.maze_width = self.COLS * self.CELL_SIZE  # 125m
        self.maze_height = self.ROWS * self.CELL_SIZE  # 125m
        
        # Başlangıç ve bitiş pozisyonları (map frame)
        # Labirent merkez (0,0), sol üst köşe başlangıç
        self.start_x = -self.maze_width / 2 + self.CELL_SIZE / 2  # -62.5 + 2.5 = -60.0
        self.start_y = -self.maze_height / 2 + self.CELL_SIZE / 2  # -62.5 + 2.5 = -60.0
        self.goal_x = self.maze_width / 2 - self.CELL_SIZE / 2  # 62.5 - 2.5 = 60.0
        self.goal_y = self.maze_height / 2 - self.CELL_SIZE / 2  # 62.5 - 2.5 = 60.0
        self.z_height = -1.5  # Labirent içinde uçmak için
        
        # QoS profilleri
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Nav2 Action Client
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            qos_profile
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            qos_profile
        )
        
        # Publishers (PX4 kontrol için)
        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos_profile
        )
        
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos_profile
        )
        
        self.command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            qos_profile
        )
        
        # State variables
        self.map_received = False
        self.map_data = None
        self.current_pose = None
        self.nav_goal_handle = None
        self.nav_velocity = None  # Nav2'den gelen velocity komutları
        self.nav_path = None  # Nav2 planı
        
        # Nav2 cmd_vel subscriber (Nav2 controller'ın gönderdiği velocity komutlarını dinle)
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            qos_profile
        )
        
        # Nav2 plan subscriber (plan varsa direkt waypoint takibi yapabiliriz)
        self.plan_sub = self.create_subscription(
            Path,
            '/plan',
            self.plan_callback,
            qos_profile
        )
        
        # Timer
        self.create_timer(0.1, self.offboard_heartbeat)
        self.create_timer(0.1, self.publish_nav2_velocity)  # Nav2 velocity'yi PX4'e çevir
        
        self.get_logger().info("Maze Navigator node started")
        self.get_logger().info(f"Labirent boyutu: {self.maze_width}m x {self.maze_height}m")
        self.get_logger().info(f"Başlangıç: ({self.start_x:.2f}, {self.start_y:.2f})")
        self.get_logger().info(f"Bitiş: ({self.goal_x:.2f}, {self.goal_y:.2f})")
        
    def map_callback(self, msg):
        """RTAB-Map haritası geldiğinde çağrılır"""
        if not self.map_received:
            self.get_logger().info("✅ RTAB-Map haritası alındı!")
            self.get_logger().info(f"Harita boyutu: {msg.info.width}x{msg.info.height}")
            self.get_logger().info(f"Çözünürlük: {msg.info.resolution}m/pixel")
            self.get_logger().info(f"Frame: {msg.header.frame_id}")
        self.map_received = True
        self.map_data = msg
        
    def odom_callback(self, msg):
        """Drone pozisyonunu güncelle"""
        self.current_pose = msg.pose.pose
        
    def cmd_vel_callback(self, msg):
        """Nav2 controller'ın gönderdiği velocity komutlarını al"""
        self.nav_velocity = msg
        
    def plan_callback(self, msg):
        """Nav2 planını al"""
        if len(msg.poses) > 0:
            self.nav_path = msg
            if not hasattr(self, '_plan_logged'):
                self.get_logger().info(f"📋 Nav2 planı alındı: {len(msg.poses)} waypoint")
                self._plan_logged = True
        
    def cmd_vel_callback(self, msg):
        """Nav2 controller'ın gönderdiği velocity komutlarını al"""
        self.nav_velocity = msg
        
    def offboard_heartbeat(self):
        """PX4 offboard mode için heartbeat gönder"""
        msg = OffboardControlMode()
        msg.position = False  # Nav2 velocity kullanıyoruz
        msg.velocity = True   # Velocity control aktif
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.offboard_pub.publish(msg)
        
    def publish_nav2_velocity(self):
        """Nav2'den gelen velocity komutlarını PX4'e çevir ve gönder"""
        if self.nav_velocity is None:
            # İlk kez velocity gelmediyse log gönder (sadece bir kez)
            if not hasattr(self, '_vel_warned'):
                self.get_logger().warn("⚠️  Nav2 velocity komutu henüz gelmedi...")
                self._vel_warned = True
            return
            
        # Nav2 cmd_vel -> PX4 TrajectorySetpoint (velocity)
        msg = TrajectorySetpoint()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.position = [float('nan')] * 3  # Position kullanmıyoruz
        
        # Nav2 2D, sadece x,y velocity gönderir
        vx = float(self.nav_velocity.linear.x)
        vy = float(self.nav_velocity.linear.y)
        
        # Z kontrolü: Hedef yüksekliği koru (Nav2 Z kontrolü yapmaz)
        if self.current_pose:
            current_z = self.current_pose.position.z
            target_z = self.z_height
            z_error = target_z - current_z
            # PID benzeri basit kontrol
            vz = float(np.clip(z_error * 0.5, -0.5, 0.5))  # Max 0.5 m/s vertical
        else:
            vz = 0.0
            
        msg.velocity = [vx, vy, vz]
        msg.yaw = float('nan')  # Yaw kontrolü Nav2'de yok
        
        self.setpoint_pub.publish(msg)
        
    def teleport_drone(self, x, y, z):
        """Drone'u belirtilen pozisyona teleport et"""
        cmd = (
            f"gz service -s /world/default/set_pose "
            f"--reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 2000 "
            f"--req 'name: \"x500_mono_cam_0\", "
            f"position: {{x: {x}, y: {y}, z: {z}}}, "
            f"orientation: {{x: 0, y: 0, z: 0, w: 1}}'"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            self.get_logger().info(f"✅ Drone teleport edildi: ({x:.2f}, {y:.2f}, {z:.2f})")
            return True
        else:
            self.get_logger().error(f"❌ Teleport başarısız: {result.stderr}")
            return False
            
    def arm_drone(self):
        """Drone'u arm et"""
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0  # Arm
        msg.param2 = 21196.0
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        
        for _ in range(10):
            self.command_pub.publish(msg)
            time.sleep(0.1)
        self.get_logger().info("✅ Arm komutu gönderildi")
        
    def set_offboard_mode(self):
        """Offboard mode'a geç"""
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0
        msg.param2 = 6.0  # OFFBOARD
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        
        for _ in range(10):
            self.command_pub.publish(msg)
            time.sleep(0.1)
        self.get_logger().info("✅ Offboard mode komutu gönderildi")
        
    def wait_for_map(self, timeout=60.0):
        """RTAB-Map haritasının gelmesini bekle"""
        self.get_logger().info("⏳ RTAB-Map haritası bekleniyor...")
        start_time = time.time()
        
        while not self.map_received:
            if time.time() - start_time > timeout:
                self.get_logger().error(f"❌ Harita timeout ({timeout}s)")
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
            
        self.get_logger().info("✅ Harita hazır!")
        return True
        
    def wait_for_nav2(self, timeout=10.0):
        """Nav2 action server'ın hazır olmasını bekle"""
        self.get_logger().info("⏳ Nav2 action server bekleniyor...")
        start_time = time.time()
        
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            if time.time() - start_time > timeout:
                self.get_logger().error(f"❌ Nav2 timeout ({timeout}s)")
                return False
            self.get_logger().info("Nav2 server bekleniyor...")
            
        self.get_logger().info("✅ Nav2 hazır!")
        return True
        
    def create_pose_stamped(self, x, y, z, yaw=0.0):
        """PoseStamped mesajı oluştur"""
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        
        # Yaw'dan quaternion'a çevir
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        pose.pose.orientation.w = cy
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = sy
        
        return pose
        
    def navigate_to_goal(self, x, y, z, yaw=0.0):
        """Nav2 ile hedefe git"""
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("❌ Nav2 server hazır değil!")
            return False
            
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.create_pose_stamped(x, y, z, yaw)
        
        self.get_logger().info(f"🎯 Hedefe gidiliyor: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        # Action gönder
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav_feedback_callback
        )
        
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error("❌ Goal kabul edilmedi!")
            return False
            
        self.get_logger().info("✅ Goal kabul edildi, rota planlanıyor...")
        self.nav_goal_handle = goal_handle
        
        # Plan'ı bekle (Nav2 plan yayınlamalı)
        self.get_logger().info("⏳ Nav2 planı bekleniyor...")
        plan_timeout = 10.0
        plan_start = time.time()
        while self.nav_path is None:
            if time.time() - plan_start > plan_timeout:
                self.get_logger().warn("⚠️  Plan timeout, Nav2 controller'a güveniyoruz...")
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if self.nav_path:
            self.get_logger().info(f"✅ Plan alındı: {len(self.nav_path.poses)} waypoint")
            # Plan varsa direkt waypoint takibi yapabiliriz (opsiyonel)
        
        # Sonucu bekle
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=300.0)  # 5 dakika timeout
        
        if not result_future.done():
            self.get_logger().error("❌ Navigation timeout!")
            goal_handle.cancel_goal_async()
            return False
            
        # Result status kontrolü
        result_wrapper = result_future.result()
        status = result_wrapper.status
        
        # ROS2 action status codes: 0=UNKNOWN, 1=ACCEPTED, 2=EXECUTING, 3=CANCELED, 4=SUCCEEDED, 5=ABORTED
        if status == 4:  # SUCCEEDED
            self.get_logger().info("✅ Hedefe ulaşıldı!")
            return True
        elif status == 3:  # CANCELED
            self.get_logger().warn("⚠️  Navigation iptal edildi")
            return False
        elif status == 5:  # ABORTED
            self.get_logger().error("❌ Navigation başarısız (aborted)!")
            return False
        else:
            self.get_logger().error(f"❌ Navigation başarısız! Status: {status}")
            return False
            
    def nav_feedback_callback(self, feedback_msg):
        """Nav2 feedback callback"""
        feedback = feedback_msg.feedback
        if self.current_pose:
            dist = math.sqrt(
                (feedback.current_pose.pose.position.x - self.goal_x)**2 +
                (feedback.current_pose.pose.position.y - self.goal_y)**2
            )
            # Her 10 feedback'te bir log (çok fazla log olmasın)
            if hasattr(self, '_fb_count'):
                self._fb_count += 1
            else:
                self._fb_count = 0
                
            if self._fb_count % 10 == 0:
                self.get_logger().info(f"📍 Kalan mesafe: {dist:.2f}m | Nav2 çalışıyor...")
            
    def run(self):
        """Ana çalıştırma fonksiyonu"""
        self.get_logger().info("=" * 60)
        self.get_logger().info("🚀 LABİRENT NAVİGASYON BAŞLIYOR")
        self.get_logger().info("=" * 60)
        
        # 1. Labirent spawn
        self.get_logger().info("\n📦 1. Labirent oluşturuluyor...")
        try:
            spawn_maze()
            self.get_logger().info("✅ Labirent oluşturuldu")
            time.sleep(2.0)  # Labirent'in spawn olması için bekle
        except Exception as e:
            self.get_logger().error(f"❌ Labirent spawn hatası: {e}")
            return False
            
        # 2. RTAB-Map haritası bekle
        self.get_logger().info("\n🗺️  2. RTAB-Map haritası bekleniyor...")
        if not self.wait_for_map(timeout=60.0):
            return False
            
        # 3. Nav2 server bekle
        self.get_logger().info("\n🧭 3. Nav2 server bekleniyor...")
        if not self.wait_for_nav2(timeout=10.0):
            return False
            
        # 4. Drone'u başlangıç pozisyonuna teleport
        self.get_logger().info("\n🚁 4. Drone başlangıç pozisyonuna götürülüyor...")
        if not self.teleport_drone(self.start_x, self.start_y, self.z_height):
            return False
        time.sleep(2.0)  # Teleport sonrası bekle
        
        # 5. Offboard mode ve arm
        self.get_logger().info("\n⚙️  5. Offboard mode ve arm...")
        self.set_offboard_mode()
        time.sleep(1.0)
        self.arm_drone()
        time.sleep(2.0)
        
        # 6. Başlangıç pozisyonunda hover (stabilizasyon) - Nav2 başlamadan önce
        self.get_logger().info("\n⏸️  6. Başlangıçta hover (stabilizasyon)...")
        # Nav2 velocity kullanacağız, bu yüzden sadece heartbeat gönder
        for _ in range(50):  # 5 saniye bekle
            self.offboard_heartbeat()
            time.sleep(0.1)
            
        # 7. Nav2 ile çıkışa git
        self.get_logger().info("\n🎯 7. Labirent çıkışına gidiliyor...")
        success = self.navigate_to_goal(
            self.goal_x,
            self.goal_y,
            self.z_height,
            yaw=0.0
        )
        
        if success:
            self.get_logger().info("\n" + "=" * 60)
            self.get_logger().info("🎉 BAŞARILI! Labirentten çıkıldı!")
            self.get_logger().info("=" * 60)
        else:
            self.get_logger().error("\n" + "=" * 60)
            self.get_logger().error("❌ BAŞARISIZ! Labirentten çıkılamadı!")
            self.get_logger().error("=" * 60)
            
        return success


def main(args=None):
    rclpy.init(args=args)
    navigator = MazeNavigator()
    
    try:
        success = navigator.run()
    except KeyboardInterrupt:
        navigator.get_logger().info("\n⚠️  Kullanıcı tarafından durduruldu")
        success = False
    except Exception as e:
        navigator.get_logger().error(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        success = False
    finally:
        navigator.destroy_node()
        rclpy.shutdown()
        
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
