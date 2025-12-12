import rclpy
from rclpy.node import Node
import numpy as np
import math
import time

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition
)

class SimpleRLPX4(Node):
    def __init__(self):
        super().__init__('simple_rl_px4')

        # -----------------------------
        # PX4 publishers
        # -----------------------------
        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            10
        )

        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            10
        )

        self.command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            10
        )

        # -----------------------------
        # PX4 subscriber
        # -----------------------------
        self.position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            10
        )

        # -----------------------------
        # State
        # -----------------------------
        self.x = 0.0
        self.y = 0.0
        self.z = -1.0
        self.position_received = False

        # -----------------------------
        # RL parameters
        # -----------------------------
        self.actions = [
            np.array([ 1.0,  0.0,  0.0]),   # forward
            np.array([-1.0,  0.0,  0.0]),   # back
            np.array([ 0.0,  1.0,  0.0]),   # left
            np.array([ 0.0, -1.0,  0.0]),   # right
            np.array([ 0.0,  0.0, -0.5]),   # up
            np.array([ 0.0,  0.0,  0.5])    # down
        ]

        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2

        # Target (very simple)
        self.goal = np.array([5.0, 0.0, -1.0])

        self.last_state = None
        self.last_action = None

        # -----------------------------
        # Timers
        # -----------------------------
        self.create_timer(0.1, self.publish_offboard_control)
        self.create_timer(0.5, self.rl_step)

        self.arm()
        self.set_offboard()

    # -------------------------------------------------
    # PX4 helpers
    # -------------------------------------------------
    def arm(self):
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            1.0
        )

    def set_offboard(self):
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            1.0,
            6.0
        )

    def send_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)

    def publish_offboard_control(self):
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    # -------------------------------------------------
    # Position callback
    # -------------------------------------------------
    def position_callback(self, msg):
        self.x = msg.x
        self.y = msg.y
        self.z = msg.z
        self.position_received = True

    # -------------------------------------------------
    # RL logic
    # -------------------------------------------------
    def get_state(self):
        return (
            round(self.x, 1),
            round(self.y, 1),
            round(self.z, 1)
        )

    def choose_action(self, state):
        if np.random.rand() < self.epsilon or state not in self.q_table:
            return np.random.randint(len(self.actions))
        return int(np.argmax(self.q_table[state]))

    def compute_reward(self):
        pos = np.array([self.x, self.y, self.z])
        dist = np.linalg.norm(pos - self.goal)
        if dist < 0.5:
            return 100.0
        return -dist

    def rl_step(self):
        if not self.position_received:
            return

        state = self.get_state()

        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

        action_idx = self.choose_action(state)
        action = self.actions[action_idx]

        # Publish velocity command
        sp = TrajectorySetpoint()
        sp.vx = action[0]
        sp.vy = action[1]
        sp.vz = action[2]
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(sp)

        reward = self.compute_reward()

        if self.last_state is not None:
            best_next = np.max(self.q_table[state])
            self.q_table[self.last_state][self.last_action] += \
                self.alpha * (reward + self.gamma * best_next -
                              self.q_table[self.last_state][self.last_action])

        self.last_state = state
        self.last_action = action_idx

        self.get_logger().info(
            f"State: {state}, Action: {action_idx}, Reward: {reward:.2f}"
        )


def main():
    rclpy.init()
    node = SimpleRLPX4()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()