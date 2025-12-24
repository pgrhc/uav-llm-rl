import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
import subprocess
import time
import threading
from px4_msgs.msg import VehicleStatus


class PX4DroneEnv(gym.Env):
    def __init__(self):
        super(PX4DroneEnv, self).__init__()

        rclpy.init(args=None)
        self.node = rclpy.create_node('px4_gym_env')

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.offboard_pub = self.node.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.setpoint_pub = self.node.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_pub = self.node.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        self.node.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos_profile)
        self.node.create_subscription(
            PointCloud2,
            '/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan/points',
            self.lidar_callback, qos_profile)
        self.node.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)

        self.current_pos = np.zeros(3)
        self.goal_pos = np.array([6.0, 6.0, -1.5])
        self.lidar_min_dist = 5.0
        self.prev_action = np.zeros(3)

        self.min_height = 0.5
        self.max_height = 2.5

        self.vehicle_status = VehicleStatus()

        self.spin_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

        time.sleep(2)

    # ---------------- CALLBACKS ----------------
    def odom_callback(self, msg):
        self.current_pos[0] = msg.position[0]
        self.current_pos[1] = msg.position[1]
        self.current_pos[2] = msg.position[2]

    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg

    def lidar_callback(self, msg):
        points = pc2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True)
        if points:
            self.lidar_min_dist = min(np.linalg.norm(p) for p in points)
        else:
            self.lidar_min_dist = 5.0

    # ---------------- STEP (AYNEN KORUNDU) ----------------
    def step(self, action):
        scale_speed = 0.8
        vx = float(action[0]) * scale_speed
        vy = float(action[1]) * scale_speed
        vz = -1.0 * float(action[2]) * scale_speed

        self.publish_velocity(vx, vy, vz)
        time.sleep(0.1)

        current_dist = np.linalg.norm(self.current_pos - self.goal_pos)
        current_height = self.current_pos[2]

        reward = -current_dist * 0.5
        terminated = False
        truncated = False

        if current_dist < 0.5:
            reward += 100.0
            terminated = True

        if self.lidar_min_dist < 0.4:
            reward -= 50.0
            terminated = True

        if current_height < self.min_height:
            reward -= 1.0
        elif current_height > self.max_height:
            reward -= 1.0

        if current_height < 0.1 or current_height > 4.0:
            reward -= 50.0
            terminated = True

        diff = np.linalg.norm(action - self.prev_action)
        reward -= diff * 0.5
        reward -= np.linalg.norm(action) * 0.1
        reward -= 0.05

        self.prev_action = action

        obs = np.array([
            self.current_pos[0],
            self.current_pos[1],
            self.current_pos[2],
            current_dist
        ], dtype=np.float32)

        return obs, reward, terminated, truncated, {}

    # ---------------- RESET (ASIL DÜZELTİLEN KISIM) ----------------
import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
import subprocess
import time
import threading
from px4_msgs.msg import VehicleStatus


class PX4DroneEnv(gym.Env):
    def __init__(self):
        super(PX4DroneEnv, self).__init__()

        rclpy.init(args=None)
        self.node = rclpy.create_node('px4_gym_env')

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.offboard_pub = self.node.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.setpoint_pub = self.node.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_pub = self.node.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        self.node.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos_profile)
        self.node.create_subscription(
            PointCloud2,
            '/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan/points',
            self.lidar_callback, qos_profile)
        self.node.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)

        self.current_pos = np.zeros(3)
        self.goal_pos = np.array([6.0, 6.0, -1.5])
        self.lidar_min_dist = 5.0
        self.prev_action = np.zeros(3)

        self.min_height = 0.5
        self.max_height = 2.5

        self.vehicle_status = VehicleStatus()

        self.spin_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

        time.sleep(2)

    # ---------------- CALLBACKS ----------------
    def odom_callback(self, msg):
        self.current_pos[0] = msg.position[0]
        self.current_pos[1] = msg.position[1]
        self.current_pos[2] = msg.position[2]

    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg

    def lidar_callback(self, msg):
        points = pc2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True)
        if points:
            self.lidar_min_dist = min(np.linalg.norm(p) for p in points)
        else:
            self.lidar_min_dist = 5.0

    # ---------------- STEP (AYNEN KORUNDU) ----------------
    def step(self, action):
        scale_speed = 0.8
        vx = float(action[0]) * scale_speed
        vy = float(action[1]) * scale_speed
        vz = -1.0 * float(action[2]) * scale_speed

        self.publish_velocity(vx, vy, vz)
        time.sleep(0.1)

        current_dist = np.linalg.norm(self.current_pos - self.goal_pos)
        current_height = self.current_pos[2]

        reward = -current_dist * 0.5
        terminated = False
        truncated = False

        if current_dist < 0.5:
            reward += 100.0
            terminated = True

        if self.lidar_min_dist < 0.4:
            reward -= 50.0
            terminated = True

        if current_height < self.min_height:
            reward -= 1.0
        elif current_height > self.max_height:
            reward -= 1.0

        if current_height < 0.1 or current_height > 4.0:
            reward -= 50.0
            terminated = True

        diff = np.linalg.norm(action - self.prev_action)
        reward -= diff * 0.5
        reward -= np.linalg.norm(action) * 0.1
        reward -= 0.05

        self.prev_action = action

        obs = np.array([
            self.current_pos[0],
            self.current_pos[1],
            self.current_pos[2],
            current_dist
        ], dtype=np.float32)

        return obs, reward, terminated, truncated, {}

    # ---------------- RESET (ASIL DÜZELTİLEN KISIM) ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("\n🔄 CLEAN RESET (ilk spawn gibi)")

        # 1️⃣ DISARM
        self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        time.sleep(0.5)

        # 2️⃣ VELOCITY SIFIRLA
        for _ in range(10):
            self.publish_velocity(0.0, 0.0, 0.0)
            self.publish_offboard_control_mode()
            time.sleep(0.1)

        # 3️⃣ DÜZ ORIENTATION + POSE
        self.teleport_drone_clean(0.0, 0.0, 0.3)
        time.sleep(0.5)

        # 4️⃣ EKF RESET
        self.send_command(VehicleCommand.VEHICLE_CMD_RESET_EKF, 1.0, 1.0)
        time.sleep(1.0)

        # 5️⃣ OFFBOARD HEARTBEAT
        for _ in range(15):
            self.publish_velocity(0.0, 0.0, 0.0)
            self.publish_offboard_control_mode()
            time.sleep(0.1)

        # 6️⃣ OFFBOARD MODE
        self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        time.sleep(0.5)

        # 7️⃣ ARM
        for _ in range(30):
            self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.publish_velocity(0.0, 0.0, 0.0)
            self.publish_offboard_control_mode()
            time.sleep(0.1)
            if self.vehicle_status.arming_state == 2:
                break

        # 8️⃣ YUMUŞAK TAKEOFF
        for _ in range(20):
            self.publish_velocity(0.0, 0.0, -0.6)
            self.publish_offboard_control_mode()
            time.sleep(0.1)

        self.prev_action = np.zeros(3)

        dist = np.linalg.norm(self.current_pos - self.goal_pos)
        obs = np.array([
            self.current_pos[0],
            self.current_pos[1],
            self.current_pos[2],
            dist
        ], dtype=np.float32)

        print("✅ Spawn temiz, kontrol RL’de\n")
        return obs, {}

    # ---------------- HELPERS ----------------
    def publish_velocity(self, vx, vy, vz):
        self.publish_offboard_control_mode()
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        msg.position = [float('nan')] * 3
        msg.velocity = [vx, vy, vz]
        msg.yaw = float('nan')
        self.setpoint_pub.publish(msg)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        self.offboard_pub.publish(msg)

    def send_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_pub.publish(msg)

    def teleport_drone_clean(self, x, y, z):
        cmd = (
            f"gz service -s /world/default/set_pose "
            f"--reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 2000 "
            f"--req 'name: \"x500_mono_cam_0\", "
            f"position: {{x: {x}, y: {y}, z: {z}}}, "
            f"orientation: {{x: 0, y: 0, z: 0, w: 1}}'"
        )
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()

    # ---------------- HELPERS ----------------
    def publish_velocity(self, vx, vy, vz):
        self.publish_offboard_control_mode()
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        msg.position = [float('nan')] * 3
        msg.velocity = [vx, vy, vz]
        msg.yaw = float('nan')
        self.setpoint_pub.publish(msg)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        self.offboard_pub.publish(msg)

    def send_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_pub.publish(msg)

    def teleport_drone_clean(self, x, y, z):
        cmd = (
            f"gz service -s /world/default/set_pose "
            f"--reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 2000 "
            f"--req 'name: \"x500_mono_cam_0\", "
            f"position: {{x: {x}, y: {y}, z: {z}}}, "
            f"orientation: {{x: 0, y: 0, z: 0, w: 1}}'"
        )
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()