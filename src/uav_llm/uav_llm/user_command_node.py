#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import rclpy
from pydantic import BaseModel, Field, ValidationError
from rclpy.node import Node
from std_msgs.msg import String


class CommandSource(str, Enum):
    TERMINAL = "terminal"


class CommandProcessingStatus(str, Enum):
    RECEIVED = "received"
    QUEUED_FOR_INTERPRETATION = "queued_for_interpretation"
    REJECTED = "rejected"


class RawUserCommand(BaseModel):
    command_id: str
    mission_id: str
    timestamp: str
    source: CommandSource
    input_mode: str = "text"
    language: str = "auto"
    user_id: str = "terminal_operator"
    raw_text: str = Field(..., min_length=1)
    status: CommandProcessingStatus


class CommandEnvelope(BaseModel):
    command_id: str
    mission_id: str
    timestamp: str
    source: CommandSource
    language: str = "auto"
    raw_text: str
    processing_status: CommandProcessingStatus


class UserCommandNode(Node):
    def __init__(self) -> None:
        super().__init__("user_command_node")

        self.declare_parameter("default_language", "auto")
        self.declare_parameter("log_to_file", True)
        self.declare_parameter(
            "log_dir",
            "/home/ubuntu/Desktop/ros2_env/uav_ws/mission_logs",
        )

        self.active_mission_id: str | None = None
        self.active_language = (
            self.get_parameter("default_language").get_parameter_value().string_value
        )
        self.log_to_file = (
            self.get_parameter("log_to_file").get_parameter_value().bool_value
        )
        self.log_dir = Path(
            self.get_parameter("log_dir").get_parameter_value().string_value
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.raw_pub = self.create_publisher(String, "/mission/user_command/raw", 10)
        self.envelope_pub = self.create_publisher(
            String, "/mission/user_command/envelope", 10
        )

        self.get_logger().info("user_command_node started.")
        self._print_welcome()

        self._input_thread = threading.Thread(target=self._terminal_loop, daemon=True)
        self._input_thread.start()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).astimezone().isoformat()

    def _generate_command_id(self) -> str:
        return f"cmd_{uuid.uuid4().hex[:10]}"

    def _generate_mission_id(self) -> str:
        return f"mission_{uuid.uuid4().hex[:6]}"

    def _publish_json(self, publisher, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        publisher.publish(msg)

    def _append_jsonl(self, filename: str, payload: dict) -> None:
        if not self.log_to_file:
            return
        path = self.log_dir / filename
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _print_welcome(self) -> None:
        print(
            "\nGörev komutlarını doğrudan yazabilirsiniz.\n"
            'Örnek: "Kırmızı kapaklı kitabı bul"\n'
            "Ek komutlar: /new, /status, /help, /reset, /quit\n"
        )

    def _print_help(self) -> None:
        print(
            "\nKullanım:\n"
            "  Doğrudan görev komutu yazabilirsiniz.\n"
            '  Örnek: "Önce 207\'ye bak sonra eski plana dön"\n\n'
            "Ek komutlar:\n"
            "  /new      -> yeni görev başlatır ve aktif yapar\n"
            "  /status   -> aktif görev bilgisini gösterir\n"
            "  /reset    -> aktif görevi temizler\n"
            "  /help     -> bu yardımı gösterir\n"
            "  /quit     -> node'u kapatır\n"
        )

    def _print_status(self) -> None:
        mission_text = self.active_mission_id if self.active_mission_id else "None"
        self.get_logger().info(
            f"Aktif durum | mission_id={mission_text} | language={self.active_language}"
        )

    def _handle_system_command(self, text: str) -> bool:
        if not text.startswith("/"):
            return False

        cmd = text.strip().lower()

        if cmd == "/help":
            self._print_help()
            return True

        if cmd == "/status":
            self._print_status()
            return True

        if cmd == "/new":
            self.active_mission_id = self._generate_mission_id()
            self.get_logger().info(
                f"Yeni görev oluşturuldu ve aktif yapıldı: {self.active_mission_id}"
            )
            return True

        if cmd == "/reset":
            old_mission = self.active_mission_id
            self.active_mission_id = None
            self.get_logger().info(
                f"Aktif görev temizlendi. Önceki görev: {old_mission}"
            )
            return True

        if cmd == "/quit":
            self.get_logger().info("Çıkış komutu alındı. Node kapatılıyor...")
            rclpy.shutdown()
            return True

        self.get_logger().warn(f"Bilinmeyen komut: {cmd}. Yardım için /help yaz.")
        return True

    def _ensure_active_mission(self) -> str:
        if self.active_mission_id is None:
            self.active_mission_id = self._generate_mission_id()
            self.get_logger().info(
                f"Aktif görev yoktu. Yeni görev otomatik oluşturuldu: {self.active_mission_id}"
            )
        return self.active_mission_id

    def _terminal_loop(self) -> None:
        while rclpy.ok():
            try:
                user_input = input("\nKomut gir > ").strip()
            except EOFError:
                self.get_logger().info("EOF received. Exiting terminal loop.")
                return
            except Exception as e:
                self.get_logger().error(f"Terminal input error: {e}")
                continue

            if not user_input:
                self.get_logger().warn("Boş komut alındı, yoksayıldı.")
                continue

            if self._handle_system_command(user_input):
                continue

            mission_id = self._ensure_active_mission()
            command_id = self._generate_command_id()
            timestamp = self._now_iso()
            language = self.active_language

            try:
                raw_cmd = RawUserCommand(
                    command_id=command_id,
                    mission_id=mission_id,
                    timestamp=timestamp,
                    source=CommandSource.TERMINAL,
                    language=language,
                    raw_text=user_input,
                    status=CommandProcessingStatus.RECEIVED,
                )
            except ValidationError as e:
                self.get_logger().error(f"RawUserCommand validation failed: {e}")
                continue

            try:
                envelope = CommandEnvelope(
                    command_id=command_id,
                    mission_id=mission_id,
                    timestamp=timestamp,
                    source=CommandSource.TERMINAL,
                    language=language,
                    raw_text=user_input,
                    processing_status=CommandProcessingStatus.QUEUED_FOR_INTERPRETATION,
                )
            except ValidationError as e:
                self.get_logger().error(f"CommandEnvelope validation failed: {e}")
                continue

            raw_dict = raw_cmd.model_dump()
            env_dict = envelope.model_dump()

            self._publish_json(self.raw_pub, raw_dict)
            self._publish_json(self.envelope_pub, env_dict)

            self._append_jsonl("raw_user_commands.jsonl", raw_dict)
            self._append_jsonl("command_envelopes.jsonl", env_dict)

            self.get_logger().info(
                f"Komut alındı ve kuyruğa eklendi | id={command_id} | mission={mission_id}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UserCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down user_command_node...")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()