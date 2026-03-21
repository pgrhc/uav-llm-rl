#!/usr/bin/env python3
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class NLCommandInputNode(Node):
    def __init__(self):
        super().__init__("nl_command_input_node")

        self.pub = self.create_publisher(String, "/llm/user_command_raw", 10)
        self._running = True

        self.get_logger().info("nl_command_input_node started.")
        self.get_logger().info("Type a natural language command and press Enter.")
        self.get_logger().info("Type 'exit' or 'quit' to stop.")

        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

    def _input_loop(self):
        while rclpy.ok() and self._running:
            try:
                user_text = input("\nNL Command> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                break

            if not user_text:
                continue

            if user_text.lower() in ["exit", "quit"]:
                self.get_logger().info("Shutting down input node...")
                self._running = False
                rclpy.shutdown()
                break

            msg = String()
            msg.data = user_text
            self.pub.publish(msg)

            self.get_logger().info(
                f"Published raw command to /llm/user_command_raw: {user_text}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = NLCommandInputNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()