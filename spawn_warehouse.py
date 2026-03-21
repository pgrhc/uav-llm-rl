#!/usr/bin/env python3
import subprocess
import time
import random
import os
import math
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────
WORLD_NAME = "default"
random.seed(time.time_ns())

# Warehouse model dosyası
WAREHOUSE_SDF_PATH = "/home/ubuntu/Desktop/warehouse/model.sdf"

# Actor skin / anim
SKIN_PATH = "/home/ubuntu/Desktop/gazebo_custom_models/actor_walking/walk.dae"
ANIM_PATH = "/home/ubuntu/Desktop/gazebo_custom_models/actor_walking/walk.dae"

# Kaç aktör
NUM_ACTORS = 6

# Drone dünya koordinatında burada spawn oluyor varsayımı
DRONE_WORLD_X = 0.0
DRONE_WORLD_Y = 0.0
DRONE_WORLD_Z = 0.0

# Warehouse girişinin MODEL LOKAL koordinatı
# BUNU SEN KENDİ MODELİNE GÖRE AYARLAYACAKSIN
# Eğer model orijini zaten girişse bunlar 0 kalabilir
ENTRANCE_LOCAL_X = 0.0
ENTRANCE_LOCAL_Y = 0.0
ENTRANCE_LOCAL_Z = 0.0

# Warehouse yaw (radyan)
WAREHOUSE_YAW = 0.0

# Actor'lar drone spawn noktasına çok yaklaşmasın
AVOID_ORIGIN_RADIUS = 2.0

# Actor yüksekliği
ACTOR_Z = 0.9

# ─────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────
def _file_uri(abs_path: str) -> str:
    abs_path = os.path.abspath(abs_path)
    return "file:///" + abs_path.lstrip("/")

def run_cmd(cmd: str):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print("STDOUT:", result.stdout.strip())
    if result.stderr.strip():
        print("STDERR:", result.stderr.strip())
    return result

def yaw_to_quat(yaw: float):
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return qx, qy, qz, qw

def rotate_2d(x, y, yaw):
    cx = math.cos(yaw)
    sx = math.sin(yaw)
    xr = x * cx - y * sx
    yr = x * sx + y * cx
    return xr, yr

def world_from_local(local_x, local_y, local_z=0.0):
    rx, ry = rotate_2d(local_x, local_y, WAREHOUSE_YAW)
    wx = WAREHOUSE_WORLD_X + rx
    wy = WAREHOUSE_WORLD_Y + ry
    wz = WAREHOUSE_WORLD_Z + local_z
    return wx, wy, wz

def gz_create_from_file(model_name: str, sdf_path: str, x=0.0, y=0.0, z=0.0, yaw=0.0):
    qx, qy, qz, qw = yaw_to_quat(yaw)
    cmd = (
        f'gz service -s /world/{WORLD_NAME}/create '
        f'--reqtype gz.msgs.EntityFactory '
        f'--reptype gz.msgs.Boolean '
        f'--timeout 5000 '
        f'--req \''
        f'sdf_filename: "{sdf_path}", '
        f'name: "{model_name}", '
        f'pose: {{ '
        f'position: {{ x: {x}, y: {y}, z: {z} }}, '
        f'orientation: {{ x: {qx}, y: {qy}, z: {qz}, w: {qw} }} '
        f'}}'
        f'\''
    )
    return run_cmd(cmd)

def gz_create_from_string(model_name: str, sdf_string: str):
    path = f"/tmp/{model_name}.sdf"
    with open(path, "w") as f:
        f.write(sdf_string)

    cmd = (
        f'gz service -s /world/{WORLD_NAME}/create '
        f'--reqtype gz.msgs.EntityFactory '
        f'--reptype gz.msgs.Boolean '
        f'--timeout 5000 '
        f'--req \'sdf_filename: "{path}"\''
    )
    return run_cmd(cmd)

def remove_entity(name: str, entity_type: str = "MODEL"):
    cmd = (
        f'gz service -s /world/{WORLD_NAME}/remove '
        f'--reqtype gz.msgs.Entity '
        f'--reptype gz.msgs.Boolean '
        f'--timeout 2000 '
        f'--req \'name: "{name}" type: {entity_type}\''
    )
    return run_cmd(cmd)

# ─────────────────────────────────────────────────────────────
# WAREHOUSE KONUM HESABI
# Giriş dünya koordinatında (0,0) olsun
# ─────────────────────────────────────────────────────────────
entrance_rx, entrance_ry = rotate_2d(ENTRANCE_LOCAL_X, ENTRANCE_LOCAL_Y, WAREHOUSE_YAW)

WAREHOUSE_WORLD_X = DRONE_WORLD_X - entrance_rx
WAREHOUSE_WORLD_Y = DRONE_WORLD_Y - entrance_ry
WAREHOUSE_WORLD_Z = DRONE_WORLD_Z - ENTRANCE_LOCAL_Z

# ─────────────────────────────────────────────────────────────
# ACTOR PATROL TANIMI
# Bunlar warehouse İÇİNDE lokal koordinatlar
# x1,y1 -> başlangıç ; x2,y2 -> bitiş
# Bu değerleri kendi warehouse içine göre ayarlayabilirsin
# ─────────────────────────────────────────────────────────────
@dataclass
class PatrolLine:
    x1: float
    y1: float
    x2: float
    y2: float

PATROL_LINES_LOCAL = [
    PatrolLine( 4.0,  2.0, 10.0,  2.0),
    PatrolLine( 4.0, -2.0, 10.0, -2.0),
    PatrolLine(12.0,  3.0, 12.0,  9.0),
    PatrolLine(15.0, -7.0, 15.0, -1.0),
    PatrolLine(20.0,  1.0, 26.0,  1.0),
    PatrolLine(22.0, -5.0, 28.0, -5.0),
]

# ─────────────────────────────────────────────────────────────
# GÜVENLİK KONTROLLERİ
# ─────────────────────────────────────────────────────────────
def point_distance_to_origin(x, y):
    return math.sqrt(x*x + y*y)

def line_is_safe(line: PatrolLine):
    # Başlangıç ve bitiş noktaları dünya koordinatına çevrilir
    wx1, wy1, _ = world_from_local(line.x1, line.y1, 0.0)
    wx2, wy2, _ = world_from_local(line.x2, line.y2, 0.0)

    d1 = point_distance_to_origin(wx1, wy1)
    d2 = point_distance_to_origin(wx2, wy2)

    return (d1 >= AVOID_ORIGIN_RADIUS) and (d2 >= AVOID_ORIGIN_RADIUS)

# ─────────────────────────────────────────────────────────────
# ACTOR SDF
# ─────────────────────────────────────────────────────────────
def build_actor_sdf(actor_name, skin_uri, anim_uri, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx*dx + dy*dy)

    if dist < 0.01:
        raise ValueError(f"{actor_name} için patrol hattı çok kısa")

    # Actor'ın gidiş yönü
    yaw_fwd = math.atan2(dy, dx)
    yaw_back = yaw_fwd + math.pi

    # Hız
    base_speed = random.uniform(0.8, 1.2)
    duration = dist / base_speed

    # Küçük bekleme
    wait_turn = random.uniform(0.4, 0.8)

    t0 = 0.0
    t1 = duration
    t2 = t1 + wait_turn
    t3 = t2 + duration
    t4 = t3 + wait_turn

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <actor name="{actor_name}">
    <pose>{x1} {y1} {ACTOR_Z} 0 0 0</pose>
    <skin>
      <filename>{skin_uri}</filename>
      <scale>1.0</scale>
    </skin>
    <animation name="walking">
      <filename>{anim_uri}</filename>
      <interpolate_x>true</interpolate_x>
    </animation>
    <script>
      <loop>true</loop>
      <auto_start>true</auto_start>
      <delay_start>{random.uniform(0.0, 2.0):.2f}</delay_start>
      <trajectory id="0" type="walking">
        <waypoint>
          <time>{t0:.2f}</time>
          <pose>0 0 0 0 0 {yaw_fwd:.5f}</pose>
        </waypoint>
        <waypoint>
          <time>{t1:.2f}</time>
          <pose>{dx:.3f} {dy:.3f} 0 0 0 {yaw_fwd:.5f}</pose>
        </waypoint>
        <waypoint>
          <time>{t2:.2f}</time>
          <pose>{dx:.3f} {dy:.3f} 0 0 0 {yaw_back:.5f}</pose>
        </waypoint>
        <waypoint>
          <time>{t3:.2f}</time>
          <pose>0 0 0 0 0 {yaw_back:.5f}</pose>
        </waypoint>
        <waypoint>
          <time>{t4:.2f}</time>
          <pose>0 0 0 0 0 {yaw_fwd:.5f}</pose>
        </waypoint>
      </trajectory>
    </script>
  </actor>
</sdf>
"""
    return sdf

# ─────────────────────────────────────────────────────────────
# SPAWN LOGIC
# ─────────────────────────────────────────────────────────────
def spawn_warehouse():
    if not os.path.exists(WAREHOUSE_SDF_PATH):
        print(f"❌ Warehouse SDF bulunamadı: {WAREHOUSE_SDF_PATH}")
        return False

    print("📦 Warehouse spawn ediliyor...")
    print(f"   World pose = ({WAREHOUSE_WORLD_X:.2f}, {WAREHOUSE_WORLD_Y:.2f}, {WAREHOUSE_WORLD_Z:.2f})")
    print(f"   Yaw        = {WAREHOUSE_YAW:.2f} rad")

    result = gz_create_from_file(
        model_name="warehouse_main",
        sdf_path=WAREHOUSE_SDF_PATH,
        x=WAREHOUSE_WORLD_X,
        y=WAREHOUSE_WORLD_Y,
        z=WAREHOUSE_WORLD_Z,
        yaw=WAREHOUSE_YAW
    )
    return result.returncode == 0

def spawn_actors():
    if not os.path.exists(SKIN_PATH):
        print(f"❌ Skin dosyası bulunamadı: {SKIN_PATH}")
        return
    if not os.path.exists(ANIM_PATH):
        print(f"❌ Anim dosyası bulunamadı: {ANIM_PATH}")
        return

    skin_uri = _file_uri(SKIN_PATH)
    anim_uri = _file_uri(ANIM_PATH)

    safe_lines = [line for line in PATROL_LINES_LOCAL if line_is_safe(line)]

    if len(safe_lines) < NUM_ACTORS:
        print(f"⚠️ Güvenli patrol hattı sayısı {len(safe_lines)}. {NUM_ACTORS} yerine o kadar spawn edilecek.")

    chosen = safe_lines[:NUM_ACTORS]

    for idx, line in enumerate(chosen):
        wx1, wy1, _ = world_from_local(line.x1, line.y1, 0.0)
        wx2, wy2, _ = world_from_local(line.x2, line.y2, 0.0)

        name = f"human_{idx}_{random.randint(100,999)}"
        sdf = build_actor_sdf(name, skin_uri, anim_uri, wx1, wy1, wx2, wy2)

        print(f"🚶 Actor[{idx}] spawn: {name}")
        print(f"   Start=({wx1:.2f},{wy1:.2f}) End=({wx2:.2f},{wy2:.2f})")

        gz_create_from_string(name, sdf)

def cleanup():
    remove_entity("warehouse_main", "MODEL")
    for i in range(20):
        remove_entity(f"human_{i}", "ACTOR")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Warehouse scenario başlatılıyor ===")

    # İstersen önce eski modeli temizle
    # remove_entity("warehouse_main", "MODEL")

    ok = spawn_warehouse()
    if not ok:
        print("❌ Warehouse spawn başarısız.")
        raise SystemExit(1)

    time.sleep(1.5)
    spawn_actors()

    print("✅ Tamamlandı.")