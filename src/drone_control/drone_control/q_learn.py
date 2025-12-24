import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import subprocess
import time
from geometry_msgs.msg import Pose

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleOdometry
)

class SimpleRLPX4(Node):
    def __init__(self):
        super().__init__('simple_rl_px4')

        # PX4 örneklerinde genelde sensor out: BEST_EFFORT + VOLATILE
        # setpoint/command in: BEST_EFFORT genelde çalışır; durability çoğunlukla VOLATILE daha güvenli
        qos_in = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        qos_out = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
                # ---------------- Maze reset ayarları ----------------
        self.maze_wall_prefix = "wall_"
        self.maze_exit_name = "maze_exit"
        self.maze_spawn_script = "/home/ubuntu/Downloads/deneme/maze_deneme.py"  # ⚠️ burayı düzelt
        # ---------------- Drone reset ayarları ----------------
        self.reset_pose = {
            "x": -5.0,
            "y": -5.0,
            "z": -1.0
        }

        self.drone_entity_name = "x500_mono_cam_0"  # Gazebo model adı

        # Publishers
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_in)

        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_in)

        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_in)

        # Subscriber
        self.position_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.position_callback,
            qos_out
        )
        # ---------------- LiDAR ----------------
        self.lidar_min_dist = None
        self.lidar_collision_threshold = 0.6  # metre (maze için ideal)

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan/points',
            self.lidar_callback,
            qos_out
        )

        # State
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.position_received = False

        # Offboard/arm state
        self.setpoints_sent = 0
        self.offboard_armed = False

        # RL actions
        self.actions = [
            np.array([ 0.5,  0.0,  0.0]),
            np.array([-0.5,  0.0,  0.0]),
            np.array([ 0.0,  0.5,  0.0]),
            np.array([ 0.0, -0.5,  0.0]),
            np.array([ 0.0,  0.0, -0.2]),  # NED: vz negatif -> yukarı
            np.array([ 0.0,  0.0,  0.5])
        ]
        # ---------------- RL parametreleri ----------------
        self.goal = np.array([6.0, 6.0, -1.0])   # 🎯 Maze çıkışı
        self.prev_dist = None
        # ---------------- Spawn (initial pose) ----------------
        self.initial_pose_saved = False
        self.initial_pose = {
            "x": None,
            "y": None,
            "z": None
}

        self.collision_penalty = -50.0
        self.goal_reward = 100.0
        self.step_penalty = -0.1

        # “Şu an basılacak hız” (setpoint timer bunu publish edecek)
        self.desired_v = np.array([0.0, 0.0, 0.0], dtype=float)

        # Timers
        self.create_timer(0.1, self.publish_offboard_and_setpoint)  # 10 Hz (çok daha stabil)
        self.create_timer(0.5, self.rl_step)                        # RL kararı daha yavaş olabilir

    # ---------------- PX4 commands ----------------
    def remove_entity(self, name):
        cmd = (
            f"gz service -s /world/default/remove "
            f"--reqtype gz.msgs.Entity "
            f"--reptype gz.msgs.Boolean "
            f"--timeout 1000 "
            f"--req 'name: \"{name}\"'"
        )
        subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def clear_maze(self):
        for i in range(300):  # güvenli üst sınır
            self.remove_entity(f"{self.maze_wall_prefix}{i}")

        self.remove_entity(self.maze_exit_name)
        self.get_logger().info("🧹 Maze temizlendi")

    def respawn_maze(self):
        self.get_logger().info("🧱 Maze yeniden spawn ediliyor")
        subprocess.run(
            ["python3", self.maze_spawn_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(1.0)  # Gazebo’nun toparlanması için

    def reset_drone_pose(self):
        if not self.initial_pose_saved:
            self.get_logger().warn("Initial pose henüz kaydedilmedi, reset atlandı")
            return

        self.get_logger().info("🚁 Drone initial spawn pozisyonuna teleport ediliyor")

        cmd = (
            f"gz service -s /world/default/set_pose "
            f"--reqtype gz.msgs.Pose "
            f"--reptype gz.msgs.Boolean "
            f"--timeout 1000 "
            f"--req 'name: \"{self.drone_entity_name}\", "
            f"position: {{ "
            f"x: {self.initial_pose['x']}, "
            f"y: {self.initial_pose['y']}, "
            f"z: {self.initial_pose['z']} "
            f"}}'"
        )

        subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        time.sleep(0.5)

        # RL state reset
        self.prev_dist = None
        self.desired_v[:] = 0.0

        # PX4 offboard tekrar aktif
        self.set_offboard()
        self.arm()

    def arm(self):
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("ARM gönderildi")
    
    def lidar_callback(self, msg: PointCloud2):
        min_dist = float('inf')

        # PointCloud2 -> (x,y,z)
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p
            d = np.sqrt(x*x + y*y + z*z)
            if d < min_dist:
                min_dist = d

        if min_dist < float('inf'):
            self.lidar_min_dist = min_dist

    def set_offboard(self):
        # PX4 örnekleri: param1=1, param2=6 -> OFFBOARD
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.get_logger().info("OFFBOARD mode komutu gönderildi")

    def send_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)

    # ------------- publish loop (10 Hz) -------------
    def publish_offboard_and_setpoint(self):
        # 1) OffboardControlMode her zaman akmalı
        ctrl = OffboardControlMode()
        ctrl.position = False
        ctrl.velocity = True
        ctrl.acceleration = False
        ctrl.attitude = False
        ctrl.body_rate = False
        ctrl.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(ctrl)

        # 2) TrajectorySetpoint her zaman akmalı (stream şart)
        sp = TrajectorySetpoint()

        # TrajectorySetpoint mesajında array alanları varsa bu doğru:
        sp.position = [float('nan'), float('nan'), float('nan')]
        sp.acceleration = [float('nan'), float('nan'), float('nan')]
        sp.yaw = float('nan')
        sp.yawspeed = float('nan')
        sp.velocity = [float(self.desired_v[0]), float(self.desired_v[1]), float(self.desired_v[2])]

        sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(sp)

        # kaç setpoint gönderildi say
        self.setpoints_sent += 1

        # 3) Offboard + arm: ancak odometry geldiyse ve bir miktar setpoint yayınlandıysa
        if (not self.offboard_armed) and self.position_received and self.setpoints_sent > 30:
            # ~3 saniye 10Hz setpoint -> PX4 artık stream görüyor
            self.set_offboard()
            self.arm()
            self.offboard_armed = True

    # ---------------- callbacks ----------------
    def position_callback(self, msg: VehicleOdometry):
        self.x = float(msg.position[0])
        self.y = float(msg.position[1])
        self.z = float(msg.position[2])
        self.position_received = True
        if not self.initial_pose_saved:
            self.initial_pose["x"] = self.x
            self.initial_pose["y"] = self.y
            self.initial_pose["z"] = self.z

            self.initial_pose_saved = True

            self.get_logger().info(
                f"📍 Initial spawn pose kaydedildi: "
                f"x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}"
            )
    
    def distance_to_goal(self):
        pos_xy = np.array([self.x, self.y])
        goal_xy = np.array([self.goal[0], self.goal[1]])
        return np.linalg.norm(pos_xy - goal_xy)

    def check_collision(self):
        if self.lidar_min_dist is None:
            return False

        if self.lidar_min_dist < self.lidar_collision_threshold:
            self.get_logger().warn(
                f"LiDAR COLLISION! min_dist={self.lidar_min_dist:.2f} m"
            )
            return True

        return False

    def compute_reward(self):
        reward = self.step_penalty
        done = False

        # --- 2D distance (main objective) ---
        dist = self.distance_to_goal()

        if self.prev_dist is None:
            self.prev_dist = dist
            return reward, done

        # --- Progress reward (XY only) ---
        reward += (self.prev_dist - dist) * 10.0

        # --- Height stabilization penalty ---
        reward += -abs(self.z - self.goal[2]) * 0.5

        # --- Goal reached ---
        if dist < 0.5:
            reward += self.goal_reward
            done = True

        # --- Collision ---
        if self.check_collision():
            reward += self.collision_penalty
            done = True

        self.prev_dist = dist
        return reward, done

    # ---------------- RL step ----------------
    def rl_step(self):
        if not self.position_received or not self.offboard_armed:
            return

        state = (round(self.x, 1), round(self.y, 1), round(self.z, 1))

        # Kalkış
        if self.z > -1.0:
            self.desired_v = np.array([0.0, 0.0, -0.8], dtype=float)
            self.get_logger().info(f"Kalkış | z={self.z:.2f}")
            return

        # Random policy (şimdilik)
        action = self.actions[np.random.randint(len(self.actions))]
        self.desired_v = action.astype(float)

        reward, done = self.compute_reward()

        self.get_logger().info(
            f"State={state} | v={self.desired_v} | reward={reward:.2f}"
        )
        self.get_logger().info(
            f"LiDAR min dist = {self.lidar_min_dist}"
        )

        if done:
            self.get_logger().warn("🔁 Episode bitti → Maze resetleniyor")

            self.prev_dist = None
            self.desired_v[:] = 0.0
            self.reset_drone_pose()
            self.clear_maze()
            self.respawn_maze()

def main():
    rclpy.init()
    node = SimpleRLPX4()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()