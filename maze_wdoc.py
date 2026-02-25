import subprocess
import time
import random
import os
import math
import json
from dataclasses import dataclass

# --- AYARLAR ---
WORLD_NAME = "default"
WALL_HEIGHT = 7
WALL_THICKNESS = 0.2
CELL_SIZE = 5.0
Z_CENTER = WALL_HEIGHT / 2

ROWS, COLS = 15, 15
random.seed(time.time_ns())

# Labirentteki hücrelerin % kaçı kadar insan (çok yüksekse performans düşer)
ACTOR_DENSITY = 0.30

# Drone Spawn Merkezi (grid hücre)
DRONE_SPAWN_CELL = (ROWS // 2, COLS // 2)

# İnsanların spawn noktasına yaklaşmaması gereken mesafe (hücre cinsinden)
AVOID_SPAWN_RADIUS_CELLS = 2.0

# --- COLLIDER AYARLARI ---
COLLIDER_RADIUS = 0.30   # m
COLLIDER_HEIGHT = 1.70   # m
COLLIDER_Z = COLLIDER_HEIGHT / 2.0
FOLLOW_HZ = 20.0         # 20 Hz takip
FOLLOW_SLEEP = 1.0 / FOLLOW_HZ

# --- DOSYA YOLLARI ---
SKIN_PATH = "/home/ubuntu/Desktop/gazebo_custom_models/actor_walking/walk.dae"
ANIM_PATH = "/home/ubuntu/Desktop/gazebo_custom_models/actor_walking/walk.dae"

WALLS_SAVE_PATH = "/home/ubuntu/Desktop/maze_walls.json"

def _file_uri(abs_path: str) -> str:
    abs_path = os.path.abspath(abs_path)
    return "file:///" + abs_path.lstrip("/")

# ---- Gazebo helper: run gz command ----
def _run(cmd: str, timeout_ms=5000, fatal=False):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if fatal and res.returncode != 0:
        raise RuntimeError(f"[CMD FAIL]\ncmd: {cmd}\nstdout:{res.stdout}\nstderr:{res.stderr}")
    return res

# ---- Temel Fonksiyonlar ----
def spawn_sdf_model(model_name: str, sdf_string: str):
    path = f"/tmp/{model_name}.sdf"
    with open(path, "w") as f:
        f.write(sdf_string)

    cmd = (
        f"gz service -s /world/{WORLD_NAME}/create "
        f"--reqtype gz.msgs.EntityFactory "
        f"--reptype gz.msgs.Boolean "
        f"--timeout 5000 "
        f"--req 'sdf_filename: \"{path}\"'"
    )
    res = _run(cmd)
    # Spawn başarısızsa en azından stderr görünsün
    if res.returncode != 0 or ("true" not in (res.stdout or "").lower()):
        print(f"[SPAWN WARN] {model_name}\nstdout:{res.stdout}\nstderr:{res.stderr}")
    else:
        print(f"✅ Spawned: {model_name}")

def remove_entity(name: str, entity_type: str = "MODEL"):
    cmd = (
        f"gz service -s /world/{WORLD_NAME}/remove "
        f"--reqtype gz.msgs.Entity "
        f"--reptype gz.msgs.Boolean "
        f"--timeout 2000 "
        f"--req 'name: \"{name}\" type: {entity_type}'"
    )
    _run(cmd)

# ---- Maze Algoritmaları ----
DIRS = {"N": (-1,0), "E": (0,1), "S": (1,0), "W": (0,-1)}
OPP  = {"N":"S","S":"N","E":"W","W":"E"}

def in_bounds(r,c,rows,cols):
    return 0 <= r < rows and 0 <= c < cols

def generate_perfect_maze(rows, cols):
    walls = [[{"N": True, "E": True, "S": True, "W": True} for _ in range(cols)] for _ in range(rows)]
    vis = [[False]*cols for _ in range(rows)]
    stack = [(0,0)]
    vis[0][0] = True

    while stack:
        r,c = stack[-1]
        neigh = []
        for d,(dr,dc) in DIRS.items():
            rr,cc = r+dr, c+dc
            if in_bounds(rr,cc,rows,cols) and not vis[rr][cc]:
                neigh.append((d,rr,cc))
        if not neigh:
            stack.pop()
            continue
        d,rr,cc = random.choice(neigh)
        walls[r][c][d] = False
        walls[rr][cc][OPP[d]] = False
        vis[rr][cc] = True
        stack.append((rr,cc))

    # sadece giriş/çıkış
    walls[0][0]["N"] = False
    walls[rows-1][cols-1]["S"] = False

    # Drone spawn hücresini “kavşak” yap (senin mantığın)
    sr, sc = DRONE_SPAWN_CELL
    if in_bounds(sr, sc, rows, cols):
        walls[sr][sc]["N"] = False
        walls[sr][sc]["S"] = False
        walls[sr][sc]["E"] = False
        walls[sr][sc]["W"] = False
        if in_bounds(sr-1, sc, rows, cols): walls[sr-1][sc]["S"] = False
        if in_bounds(sr+1, sc, rows, cols): walls[sr+1][sc]["N"] = False
        if in_bounds(sr, sc-1, rows, cols): walls[sr][sc-1]["E"] = False
        if in_bounds(sr, sc+1, rows, cols): walls[sr][sc+1]["W"] = False

    return walls

@dataclass
class Segment:
    x: float
    y: float
    length: float
    horizontal: bool

def maze_to_segments(walls, rows, cols, cell_size):
    # Drone spawn (merkez) noktasına göre labirenti kaydır (0,0 drone hücresinin merkezi)
    spawn_r, spawn_c = DRONE_SPAWN_CELL
    ox = - (spawn_c + 0.5) * cell_size
    oy = - (spawn_r + 0.5) * cell_size

    h = [[False]*cols for _ in range(rows+1)]
    v = [[False]*(cols+1) for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if walls[r][c]["N"]: h[r][c] = True
            if walls[r][c]["S"]: h[r+1][c] = True
            if walls[r][c]["W"]: v[r][c] = True
            if walls[r][c]["E"]: v[r][c+1] = True

    segs = []

    # merge horizontal
    for r in range(rows+1):
        c = 0
        while c < cols:
            if not h[r][c]:
                c += 1; continue
            start = c
            while c < cols and h[r][c]:
                c += 1
            end = c
            length = (end-start)*cell_size
            x = ox + (start*cell_size + end*cell_size)/2.0
            y = oy + r*cell_size
            segs.append(Segment(x,y,length,True))

    # merge vertical
    for c in range(cols+1):
        r = 0
        while r < rows:
            if not v[r][c]:
                r += 1; continue
            start = r
            while r < rows and v[r][c]:
                r += 1
            end = r
            length = (end-start)*cell_size
            x = ox + c*cell_size
            y = oy + (start*cell_size + end*cell_size)/2.0
            segs.append(Segment(x,y,length,False))

    return segs

def build_maze_sdf(model_name: str, segments):
    links = []
    for i, seg in enumerate(segments):
        sx, sy = (seg.length, WALL_THICKNESS) if seg.horizontal else (WALL_THICKNESS, seg.length)
        links.append(f"""
    <link name="wall_{i}">
      <pose>{seg.x} {seg.y} {Z_CENTER} 0 0 0</pose>
      <collision name="c">
        <geometry><box><size>{sx} {sy} {WALL_HEIGHT}</size></box></geometry>
      </collision>
      <visual name="v">
        <geometry><box><size>{sx} {sy} {WALL_HEIGHT}</size></box></geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
        </material>
      </visual>
    </link>""")
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{model_name}">
    <static>true</static>
    {''.join(links)}
  </model>
</sdf>"""

def save_walls(walls, path=WALLS_SAVE_PATH):
    with open(path, "w") as f:
        json.dump(walls, f)
    print(f"✅ Walls kaydedildi: {path}")

# ---- Actor Logic ----
def collect_corridors(walls, rows, cols, min_len=2):
    corridors = []

    # Yatay
    for r in range(rows):
        c = 0
        while c < cols:
            if c < cols-1 and not walls[r][c]["E"]:
                start = c
                while c < cols-1 and not walls[r][c]["E"]:
                    c += 1
                length = c - start + 1
                if length >= min_len:
                    corridors.append(((r, start), (r, c), True))
            c += 1

    # Dikey
    for c in range(cols):
        r = 0
        while r < rows:
            if r < rows-1 and not walls[r][c]["S"]:
                start = r
                while r < rows-1 and not walls[r][c]["S"]:
                    r += 1
                length = r - start + 1
                if length >= min_len:
                    corridors.append(((start, c), (r, c), False))
            r += 1

    return corridors

def corridor_to_world(corridor, rows, cols, cell_size):
    (r1, c1), (r2, c2), is_horiz = corridor
    spawn_r, spawn_c = DRONE_SPAWN_CELL
    ox = - (spawn_c + 0.5) * cell_size
    oy = - (spawn_r + 0.5) * cell_size

    x1 = ox + (c1 + 0.5) * cell_size
    y1 = oy + (r1 + 0.5) * cell_size
    x2 = ox + (c2 + 0.5) * cell_size
    y2 = oy + (r2 + 0.5) * cell_size
    return x1, y1, x2, y2, is_horiz

def pick_actor_count(rows, cols):
    area = rows * cols
    n = max(1, int(area * ACTOR_DENSITY))
    return min(n, 150)

def _corridor_near_spawn(corridor, spawn_r, spawn_c, radius):
    (r1, c1), (r2, c2), _ = corridor
    if (abs(r1 - spawn_r) < radius and abs(c1 - spawn_c) < radius) or \
       (abs(r2 - spawn_r) < radius and abs(c2 - spawn_c) < radius):
        return True
    return False

def build_actor_sdf(actor_name, skin_uri, anim_uri, x1, y1, x2, y2, is_horiz):
    yaw_fwd = 0.0 if is_horiz else 1.57
    yaw_back = 3.14 if is_horiz else -1.57

    dx = x2 - x1
    dy = y2 - y1

    ACTOR_Z = 0.9
    dist = (dx*dx + dy*dy) ** 0.5

    # hız profili (rastgele)
    base_speed = random.choice([0.6, 1.0, 1.4])  # m/s
    duration = max(1.0, dist / base_speed)

    turn_wait = random.choice([0.2, 0.5, 0.8])

    t0 = 0.0
    t1 = duration
    t2 = t1 + turn_wait
    t3 = t2 + duration
    t4 = t3 + turn_wait

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <actor name="{actor_name}">
    <pose>{x1} {y1} {ACTOR_Z} 0 0 0</pose>
    <skin><filename>{skin_uri}</filename><scale>1.0</scale></skin>
    <animation name="walking"><filename>{anim_uri}</filename><interpolate_x>true</interpolate_x></animation>
    <script>
      <loop>true</loop>
      <auto_start>true</auto_start>
      <trajectory id="0" type="walking">
        <waypoint><time>{t0:.2f}</time><pose>0 0 0 0 0 {yaw_fwd}</pose></waypoint>
        <waypoint><time>{t1:.2f}</time><pose>{dx:.3f} {dy:.3f} 0 0 0 {yaw_fwd}</pose></waypoint>
        <waypoint><time>{t2:.2f}</time><pose>{dx:.3f} {dy:.3f} 0 0 0 {yaw_back}</pose></waypoint>
        <waypoint><time>{t3:.2f}</time><pose>0 0 0 0 0 {yaw_back}</pose></waypoint>
        <waypoint><time>{t4:.2f}</time><pose>0 0 0 0 0 {yaw_fwd}</pose></waypoint>
      </trajectory>
    </script>
  </actor>
</sdf>
"""
    # collider takip için meta (world koordinatları + süreler)
    meta = {
        "actor": actor_name,
        "x1": x1, "y1": y1,
        "x2": x2, "y2": y2,
        "t_go": duration,
        "t_turn": turn_wait,
        "is_horiz": is_horiz,
        "yaw_fwd": yaw_fwd,
        "yaw_back": yaw_back,
    }
    return sdf, meta

# ---- COLLIDER: model SDF ----
def build_collider_sdf(model_name: str, x: float, y: float, yaw: float):
    # görünmez collision silindir (LiDAR/Radar/çarpışma için)
    # kinematic: dışarıdan pose basacağız
    # visual yok (istersen debug için ekleyebilirsin)
    r = COLLIDER_RADIUS
    h = COLLIDER_HEIGHT
    z = COLLIDER_Z
    # basit inertia
    mass = 30.0
    ixx = (1/12) * mass * (3*r*r + h*h)
    iyy = ixx
    izz = 0.5 * mass * r*r

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{model_name}">
    <static>false</static>
    <link name="link">
      <pose>{x} {y} {z} 0 0 {yaw}</pose>
      <kinematic>true</kinematic>
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx><iyy>{iyy}</iyy><izz>{izz}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="col">
        <geometry>
          <cylinder>
            <radius>{r}</radius>
            <length>{h}</length>
          </cylinder>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""
    return sdf

# ---- set_pose service discovery + pose set ----
def _find_set_pose_service():
    res = _run("gz service -l")
    if res.returncode != 0:
        print("❌ gz service -l çalışmadı.")
        print(res.stderr)
        return None

    lines = (res.stdout or "").splitlines()
    # en çok görülen isim: /world/<world>/set_pose
    preferred = f"/world/{WORLD_NAME}/set_pose"
    for ln in lines:
        if ln.strip() == preferred:
            return preferred

    # fallback: içinde set_pose geçen world servisi ara
    cand = []
    for ln in lines:
        s = ln.strip()
        if f"/world/{WORLD_NAME}/" in s and "set_pose" in s:
            cand.append(s)
    return cand[0] if cand else None

def _yaw_to_quat(yaw: float):
    # roll=pitch=0
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    # w, x, y, z
    return cy, 0.0, 0.0, sy

def set_model_pose(service_name: str, model_name: str, x: float, y: float, z: float, yaw: float):
    w, qx, qy, qz = _yaw_to_quat(yaw)
    req = (
        f'name: "{model_name}" '
        f'position: {{x: {x:.5f}, y: {y:.5f}, z: {z:.5f}}} '
        f'orientation: {{w: {w:.6f}, x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}}}'
    )

    cmd = (
        f'gz service -s {service_name} '
        f'--reqtype gz.msgs.Pose '
        f'--reptype gz.msgs.Boolean '
        f'--timeout 1000 '
        f'--req \'{req}\''
    )
    _run(cmd)

# ---- collider follow loop ----
def _pos_on_pingpong(meta, t):
    """
    meta: x1,y1,x2,y2,t_go,t_turn,yaw_fwd,yaw_back
    t: elapsed seconds
    returns x,y,yaw
    """
    x1, y1, x2, y2 = meta["x1"], meta["y1"], meta["x2"], meta["y2"]
    t_go = meta["t_go"]
    t_turn = meta["t_turn"]
    yaw_fwd = meta["yaw_fwd"]
    yaw_back = meta["yaw_back"]

    period = 2*t_go + 2*t_turn
    if period <= 1e-6:
        return x1, y1, yaw_fwd

    tt = t % period

    # 0..t_go : ileri
    if tt < t_go:
        a = tt / t_go
        x = x1 + (x2 - x1) * a
        y = y1 + (y2 - y1) * a
        return x, y, yaw_fwd

    # t_go..t_go+t_turn : dönüş bekleme (uçta)
    if tt < t_go + t_turn:
        return x2, y2, yaw_back

    # t_go+t_turn..2*t_go+t_turn : geri
    if tt < 2*t_go + t_turn:
        a = (tt - (t_go + t_turn)) / t_go
        x = x2 + (x1 - x2) * a
        y = y2 + (y1 - y2) * a
        return x, y, yaw_back

    # son dönüş bekleme (başta)
    return x1, y1, yaw_fwd

def run_collider_follow_loop(actors_meta):
    svc = _find_set_pose_service()
    if not svc:
        print("❌ set_pose servisi bulunamadı.")
        print("İpucu: `gz service -l | grep set_pose` çıktısını kontrol et.")
        return

    print(f"✅ Collider takip servisi: {svc}")
    t0 = time.time()

    try:
        while True:
            now = time.time()
            elapsed = now - t0
            for meta in actors_meta:
                col = meta["collider"]
                x, y, yaw = _pos_on_pingpong(meta, elapsed)
                set_model_pose(svc, col, x, y, COLLIDER_Z, yaw)
            time.sleep(FOLLOW_SLEEP)
    except KeyboardInterrupt:
        print("\n⏹️ Takip döngüsü durduruldu (Ctrl+C).")

def spawn_multiple_actors(walls, rows, cols):
    if not os.path.exists(SKIN_PATH) or not os.path.exists(ANIM_PATH):
        print("❌ Skin/Anim dosyaları eksik.")
        return []

    corridors = collect_corridors(walls, rows, cols, min_len=2)

    sr, sc = DRONE_SPAWN_CELL
    safe_corridors = [c for c in corridors if not _corridor_near_spawn(c, sr, sc, AVOID_SPAWN_RADIUS_CELLS)]
    if not safe_corridors:
        print("❌ Uygun koridor bulunamadı.")
        return []

    n = min(pick_actor_count(rows, cols), len(safe_corridors))
    print(f"🎯 Hedeflenen İnsan Sayısı: {n}")

    chosen = random.sample(safe_corridors, k=n)
    skin_uri = _file_uri(SKIN_PATH)
    anim_uri = _file_uri(ANIM_PATH)

    metas = []

    for idx, corr in enumerate(chosen):
        x1, y1, x2, y2, is_horiz = corridor_to_world(corr, rows, cols, CELL_SIZE)

        actor_name = f"human_{idx}_{random.randint(100,999)}"
        collider_name = f"{actor_name}_col"

        # actor sdf + meta
        actor_sdf, meta = build_actor_sdf(actor_name, skin_uri, anim_uri, x1, y1, x2, y2, is_horiz)

        # collider sdf (başlangıçta x1,y1)
        col_sdf = build_collider_sdf(collider_name, x1, y1, meta["yaw_fwd"])

        # önce eskilerini kaldır
        remove_entity(actor_name, "ACTOR")
        remove_entity(collider_name, "MODEL")

        # spawn
        spawn_sdf_model(actor_name, actor_sdf)
        spawn_sdf_model(collider_name, col_sdf)

        # meta’ya collider ismini ekle
        meta["collider"] = collider_name
        metas.append(meta)

        print(f"✅ Actor+Collider spawn: {actor_name} + {collider_name}")

    return metas

# ---- MAIN ----
if __name__ == "__main__":
    print("🧩 Maze oluşturuluyor...")
    remove_entity("maze_current", "MODEL")

    walls = generate_perfect_maze(ROWS, COLS)
    save_walls(walls)

    segments = maze_to_segments(walls, ROWS, COLS, CELL_SIZE)
    maze_sdf = build_maze_sdf("maze_current", segments)
    spawn_sdf_model("maze_current", maze_sdf)

    time.sleep(1.0)

    actors_meta = spawn_multiple_actors(walls, ROWS, COLS)

    # Collider’ları actor rotasına kilitle (20 Hz)
    if actors_meta:
        run_collider_follow_loop(actors_meta)