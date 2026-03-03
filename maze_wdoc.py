import subprocess
import time
import random
import os
from dataclasses import dataclass

# --- AYARLAR ---
WORLD_NAME = "default"
WALL_HEIGHT = 7
WALL_THICKNESS = 0.2
CELL_SIZE = 5.0
Z_CENTER = WALL_HEIGHT / 2

ROWS, COLS = 15, 15
random.seed(time.time_ns())

# --- DÜZELTME 1: İNSAN SAYISI AYARI ---
# Labirentteki hücrelerin %15'ine insan koy (Eskiden 0.03 idi)
ACTOR_DENSITY = 0.3  
# Drone Spawn Merkezi
DRONE_SPAWN_CELL = (ROWS // 2, COLS // 2)
# İnsanların spawn noktasına yaklaşmaması gereken mesafe (hücre cinsinden)
AVOID_SPAWN_RADIUS_CELLS = 2.0

# --- DOSYA YOLLARI ---
SKIN_PATH = "/home/ubuntu/Desktop/gazebo_custom_models/actor_walking/walk.dae"
ANIM_PATH = "/home/ubuntu/Desktop/gazebo_custom_models/actor_walking/walk.dae"
box_uri = "/home/ubuntu/Desktop/gazebo_custom_models/actor_walking/Untitled.dae"

def _file_uri(abs_path: str) -> str:
    abs_path = os.path.abspath(abs_path)
    return "file:///" + abs_path.lstrip("/")

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
    subprocess.run(cmd, shell=True, capture_output=True, text=True)

def remove_entity(name: str, entity_type: str = "MODEL"):
    cmd = (
        f"gz service -s /world/{WORLD_NAME}/remove "
        f"--reqtype gz.msgs.Entity "
        f"--reptype gz.msgs.Boolean "
        f"--timeout 2000 "
        f"--req 'name: \"{name}\" type: {entity_type}'"
    )
    subprocess.run(cmd, shell=True, capture_output=True, text=True)

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

    # --- DÜZELTME 2: DUVARLARI NORMALE DÖNDÜRME ---
    # Artık sadece giriş/çıkış açık. 
    # Drone'un doğduğu yerdeki komşu duvarları silmiyoruz.
    
    walls[0][0]["N"] = False
    walls[rows-1][cols-1]["S"] = False
    
    # Sadece drone'un tam doğduğu hücreyi biraz rahatlatmak için 
    # o hücrenin kendi duvarlarını açabiliriz (Opsiyonel, kapalı kalsın dersen burayı da sil)
    # Ama komşularına (sr-1, sr+1 vb) dokunmuyoruz.
    sr, sc = DRONE_SPAWN_CELL
    if in_bounds(sr, sc, rows, cols):
        # 4 tarafı açık bir kavşak yapıyoruz (sadece o hücre için)
        walls[sr][sc]["N"] = False
        walls[sr][sc]["S"] = False
        walls[sr][sc]["E"] = False
        walls[sr][sc]["W"] = False
        
        # Karşılık gelen komşu duvarları da açmalıyız ki duvarın tek yüzü silinip diğeri kalmasın
        if in_bounds(sr-1, sc, rows, cols): walls[sr-1][sc]["S"] = False
        if in_bounds(sr+1, sc, rows, cols): walls[sr+1][sc]["N"] = False
        if in_bounds(sr, sc-1, rows, cols): walls[sr][sc-1]["E"] = False
        if in_bounds(sr, sc+1, rows, cols): walls[sr][sc+1]["W"] = False

    return walls

@dataclass
class Segment:
    x: float; y: float; length: float; horizontal: bool

def maze_to_segments(walls, rows, cols, cell_size):
    # Drone spawn (merkez) noktasına göre labirenti kaydır
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
    # Yatay Duvarlar
    for r in range(rows+1):
        c = 0
        while c < cols:
            if not h[r][c]: c += 1; continue
            start = c
            while c < cols and h[r][c]: c += 1
            length = (c - start) * cell_size
            x = ox + (start + c) * cell_size / 2.0
            y = oy + r * cell_size
            segs.append(Segment(x, y, length, True))
    # Dikey Duvarlar
    for c in range(cols+1):
        r = 0
        while r < rows:
            if not v[r][c]: r += 1; continue
            start = r
            while r < rows and v[r][c]: r += 1
            length = (r - start) * cell_size
            x = ox + c * cell_size
            y = oy + (start + r) * cell_size / 2.0
            segs.append(Segment(x, y, length, False))
    return segs

def build_maze_sdf(model_name: str, segments):
    links = []
    for i, seg in enumerate(segments):
        sx, sy = (seg.length, WALL_THICKNESS) if seg.horizontal else (WALL_THICKNESS, seg.length)
        links.append(f"""
    <link name="wall_{i}">
      <pose>{seg.x} {seg.y} {Z_CENTER} 0 0 0</pose>
      <collision name="c"><geometry><box><size>{sx} {sy} {WALL_HEIGHT}</size></box></geometry></collision>
      <visual name="v"><geometry><box><size>{sx} {sy} {WALL_HEIGHT}</size></box></geometry>
        <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material></visual>
    </link>""")
    return f"""<?xml version="1.0"?><sdf version="1.9"><model name="{model_name}"><static>true</static>{''.join(links)}</model></sdf>"""

# ---- Actor Logic (GELİŞMİŞ) ----

def collect_corridors(walls, rows, cols, min_len=2):
    """Hem yatay hem dikey koridorları bulur"""
    corridors = []
    
    # 1. Yatay Koridorlar
    for r in range(rows):
        c = 0
        while c < cols:
            if c < cols-1 and not walls[r][c]["E"]:
                start = c
                while c < cols-1 and not walls[r][c]["E"]: c += 1
                length = c - start + 1
                if length >= min_len:
                    corridors.append(((r, start), (r, c), True)) # True = Horizontal
            c += 1
            
    # 2. Dikey Koridorlar (YENİ)
    for c in range(cols):
        r = 0
        while r < rows:
            if r < rows-1 and not walls[r][c]["S"]:
                start = r
                while r < rows-1 and not walls[r][c]["S"]: r += 1
                length = r - start + 1
                if length >= min_len:
                    corridors.append(((start, c), (r, c), False)) # False = Vertical
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
    # %15 yoğunluk (Örn: 100 hücrede 15 insan)
    n = max(1, int(area * ACTOR_DENSITY))
    return min(n, 150) # Max sınırını da artırdım

def build_actor_sdf(actor_name, skin_uri, anim_uri, x1, y1, x2, y2, is_horiz):
    # Dikey ise 90 derece (1.57), Yatay ise 0 derece
    yaw_fwd = 0.0 if is_horiz else 1.57
    yaw_back = 3.14 if is_horiz else -1.57
    
    dx = x2 - x1
    dy = y2 - y1
    ACTOR_Z = 0.9
    dist = (dx*dx + dy*dy) ** 0.5
    
    # Hız Profili
    base_speed = 1.0 # m/s
    duration = dist / base_speed
    
    t0 = 0.0
    t1 = duration
    t2 = t1 + 0.5 # Dönüş bekleme
    t3 = t2 + duration
    t4 = t3 + 0.5

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
    return sdf

def build_collision_box_sdf(name, x, y, x2, y2, is_horiz):
    dx, dy = x2 - x, y2 - y
    duration = ((dx**2 + dy**2)**0.5) / 1.0
    yaw_fwd = 0.0 if is_horiz else 1.57
    yaw_back = 3.14 if is_horiz else -1.57

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}_phys">
    <static>false</static>
    <pose>{x} {y} -0.6 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>80.0</mass>
        <inertia><ixx>10</ixx><iyy>10</iyy><izz>10</izz></inertia>
      </inertial>
      <collision name="collision">
        <geometry><mesh><uri>{box_uri}</uri></mesh></geometry>
        <surface><contact><collide_bitmask>0xFF</collide_bitmask></contact></surface>
      </collision>
      <visual name="visual">
        <geometry><mesh><uri>{box_uri}</uri></mesh></geometry>
        <material><ambient>1 0 0 0.3</ambient><diffuse>1 0 0 0.3</diffuse></material>
      </visual>
    </link>
    <script>
      <loop>true</loop>
      <auto_start>true</auto_start>
      <trajectory id="0" type="walking">
        <waypoint><time>0.0</time><pose>0 0 0 0 0 {yaw_fwd}</pose></waypoint>
        <waypoint><time>{duration:.2f}</time><pose>{dx:.3f} {dy:.3f} 0 0 0 {yaw_fwd}</pose></waypoint>
        <waypoint><time>{duration+0.5:.2f}</time><pose>{dx:.3f} {dy:.3f} 0 0 0 {yaw_back}</pose></waypoint>
        <waypoint><time>{2*duration+0.5:.2f}</time><pose>0 0 0 0 0 {yaw_back}</pose></waypoint>
      </trajectory>
    </script>
  </model>
</sdf>"""
    return sdf
def _corridor_near_spawn(corridor, spawn_r, spawn_c, radius):
    (r1, c1), (r2, c2), _ = corridor
    # Basitçe koridorun herhangi bir ucu spawn'a çok yakınsa ele
    # (Daha hassas kontrol yapılabilir ama bu yeterli)
    if (abs(r1 - spawn_r) < radius and abs(c1 - spawn_c) < radius) or \
       (abs(r2 - spawn_r) < radius and abs(c2 - spawn_c) < radius):
        return True
    return False

def spawn_multiple_actors(walls, rows, cols):
    if not os.path.exists(SKIN_PATH):
        print("❌ Dosyalar eksik.")
        return

    # Hem Yatay Hem Dikey koridorları topla
    corridors = collect_corridors(walls, rows, cols, min_len=2)
    
    sr, sc = DRONE_SPAWN_CELL
    # Spawn yakını temizle
    safe_corridors = [c for c in corridors if not _corridor_near_spawn(c, sr, sc, AVOID_SPAWN_RADIUS_CELLS)]
    
    if not safe_corridors:
        print("❌ Uygun koridor bulunamadı.")
        return

    n = min(pick_actor_count(rows, cols), len(safe_corridors))
    print(f"🎯 Hedeflenen İnsan Sayısı: {n}")
    
    chosen = random.sample(safe_corridors, k=n)
    skin_uri = _file_uri(SKIN_PATH)
    anim_uri = _file_uri(ANIM_PATH)

    for idx, corr in enumerate(chosen):
        x1, y1, x2, y2, is_horiz = corridor_to_world(corr, rows, cols, CELL_SIZE)
        name = f"human_{idx}_{random.randint(100,999)}"
        sdf = build_actor_sdf(name, skin_uri, anim_uri, x1, y1, x2, y2, is_horiz)
        
        # remove_entity(name, "ACTOR")
        spawn_sdf_model(name, sdf)
        print(f"✅ Actor[{idx}] spawn edildi.")

# ─── WALLS KAYDETME (auto_maze_navigator.py için) ──────────────────────────
import json

WALLS_SAVE_PATH = "/home/ubuntu/Desktop/maze_walls.json"

def save_walls(walls, path=WALLS_SAVE_PATH):
    """
    Maze duvar verisini JSON olarak kaydet.
    auto_maze_navigator.py bu dosyayı okuyarak A* hesaplar.
    """
    with open(path, "w") as f:
        json.dump(walls, f)
    print(f"✅ Walls kaydedildi: {path}")
# ────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("🧩 Maze oluşturuluyor...")
    #remove_entity("maze_current", "MODEL")

    walls = generate_perfect_maze(ROWS, COLS)

    # ← YENİ: Walls'ı kaydet ki auto_maze_navigator okusun
    save_walls(walls)

    segments = maze_to_segments(walls, ROWS, COLS, CELL_SIZE)
    maze_sdf = build_maze_sdf("maze_current", segments)
    spawn_sdf_model("maze_current", maze_sdf)

    time.sleep(1.0)
    spawn_multiple_actors(walls, ROWS, COLS)