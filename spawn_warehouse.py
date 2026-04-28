#!/usr/bin/env python3
import subprocess
import os
import math

WORLD_NAME = "default"
SDF_PATH = "/home/ubuntu/Desktop/gazebo_custom_models/house/house/model.sdf"

MODEL_NAME = "building"

# İstersen pozisyon ver
X = 1.0
Y = 0.0
Z = 0.0
YAW = 0.0


def yaw_to_quat(yaw: float):
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return qx, qy, qz, qw


def run_cmd(cmd: str):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("RETURN CODE:", result.returncode)
    if result.stdout.strip():
        print("STDOUT:\n", result.stdout)
    if result.stderr.strip():
        print("STDERR:\n", result.stderr)
    return result


def spawn_sdf():
    if not os.path.exists(SDF_PATH):
        print(f"Dosya yok: {SDF_PATH}")
        return

    qx, qy, qz, qw = yaw_to_quat(YAW)

    cmd = (
        f'gz service -s /world/{WORLD_NAME}/create '
        f'--reqtype gz.msgs.EntityFactory '
        f'--reptype gz.msgs.Boolean '
        f'--timeout 5000 '
        f'--req \''
        f'sdf_filename: "{SDF_PATH}", '
        f'name: "{MODEL_NAME}", '
        f'pose: {{ '
        f'position: {{ x: {X}, y: {Y}, z: {Z} }}, '
        f'orientation: {{ x: {qx}, y: {qy}, z: {qz}, w: {qw} }} '
        f'}}'
        f'\''
    )

    print("Çalışan komut:")
    print(cmd)
    run_cmd(cmd)


if __name__ == "__main__":
    spawn_sdf()