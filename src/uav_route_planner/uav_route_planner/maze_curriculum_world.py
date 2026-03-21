#!/usr/bin/env python3
"""
Mega-World Maze Curriculum Script

Iki mod:
  A) Ayri mazeler (varsayilan): 3 maze farkli koordinatlarda, teleport ile gecis
  B) Birlesik maze (--unified): Tek maze, 3 bolum yan yana, fiziksel gecis

Birlesik maze (--unified):
  - Bolum 1 (x < 50):   Basit, spawn acik
  - Bolum 2 (50-125):   Dar koridorlar, statik engeller
  - Bolum 3 (x >= 125): Ayni + 180-200 dinamik insan
  - Drone bolum 1'den cikinca 2'ye, 2'den cikinca 3'e gecer (teleport yok)

Kullanim:
  maze_curriculum_world                    # 3 ayri maze
  maze_curriculum_world --unified          # Tek birlesik maze
  maze_curriculum_world --unified --actors # Birlesik + aktorler
"""

import subprocess
import time
import random
import os
import json
import math
import sys
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL AYARLAR
# ═══════════════════════════════════════════════════════════════════════════════

WORLD_NAME = "default"
WALL_HEIGHT = 15  # Yuksek duvarlar — drone ustunden ucamaz
WALL_THICKNESS = 0.2
CELL_SIZE = 5.0
Z_CENTER = WALL_HEIGHT / 2

ROWS, COLS = 15, 15
DRONE_SPAWN_CELL = (ROWS // 2, COLS // 2)

# Birlesik maze: 15x45 (3 bolum x 15 kolon)
UNIFIED_ROWS, UNIFIED_COLS = 15, 45
UNIFIED_SECTION_COLS = 15
UNIFIED_ORIGIN = (0.0, 0.0)
UNIFIED_SPAWN_CELL = (7, 7)  # Bolum 1 merkezi

# Bolum sinirlari (world x koordinati)
SECTION1_X_MAX = 50.0   # x < 50  -> Bolum 1
SECTION2_X_MAX = 125.0  # 50 <= x < 125 -> Bolum 2
# x >= 125 -> Bolum 3

STAGE_ORIGINS = {
    1: (0.0, 0.0),
    2: (1000.0, 0.0),
    3: (2000.0, 0.0),
}

ACTOR_MIN = 180
ACTOR_MAX = 200
AVOID_SPAWN_RADIUS = 1
LONG_CORRIDOR_BONUS = 4

# Farkli hizlarda hareket eden insanlar (m/s)
ACTOR_SPEED_MIN = 0.5   # Yavas yuruyus
ACTOR_SPEED_MAX = 2.0   # Hizli yuruyus / kosu

SKIN_PATH = "/home/ubuntu/Desktop/gazebo_custom_models/actor_walking/walk.dae"
ANIM_PATH = "/home/ubuntu/Desktop/gazebo_custom_models/actor_walking/walk.dae"

WALLS_DIR = "/home/ubuntu/Desktop"
WALLS_PATHS = {
    1: os.path.join(WALLS_DIR, "maze_walls_stage1.json"),
    2: os.path.join(WALLS_DIR, "maze_walls_stage2.json"),
    3: os.path.join(WALLS_DIR, "maze_walls_stage3.json"),
}
WALLS_UNIFIED_PATH = os.path.join(WALLS_DIR, "maze_walls_unified.json")
ACTORS_JSON_PATH = os.path.join(WALLS_DIR, "maze_actors_stage3.json")
ACTORS_UNIFIED_JSON_PATH = os.path.join(WALLS_DIR, "maze_actors_unified.json")


# ═══════════════════════════════════════════════════════════════════════════════
# MAZE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

DIRS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _in_bounds(r, c, rows=ROWS, cols=COLS):
    return 0 <= r < rows and 0 <= c < cols


def generate_maze(seed=None):
    rng = random.Random(seed)
    walls = [
        [{"N": True, "E": True, "S": True, "W": True} for _ in range(COLS)]
        for _ in range(ROWS)
    ]
    vis = [[False] * COLS for _ in range(ROWS)]
    stack = [(0, 0)]
    vis[0][0] = True

    while stack:
        r, c = stack[-1]
        neigh = [
            (d, r + dr, c + dc)
            for d, (dr, dc) in DIRS.items()
            if _in_bounds(r + dr, c + dc) and not vis[r + dr][c + dc]
        ]
        if not neigh:
            stack.pop()
            continue
        d, rr, cc = rng.choice(neigh)
        walls[r][c][d] = False
        walls[rr][cc][OPP[d]] = False
        vis[rr][cc] = True
        stack.append((rr, cc))

    walls[0][0]["N"] = False
    walls[ROWS - 1][COLS - 1]["S"] = False
    return walls


def open_spawn_area(walls, radius=1):
    sr, sc = DRONE_SPAWN_CELL
    for r in range(max(0, sr - radius), min(ROWS, sr + radius + 1)):
        for c in range(max(0, sc - radius), min(COLS, sc + radius + 1)):
            for d, (dr, dc) in DIRS.items():
                nr, nc = r + dr, c + dc
                if _in_bounds(nr, nc):
                    walls[r][c][d] = False
                    walls[nr][nc][OPP[d]] = False
    return walls


def generate_unified_maze(seed=101):
    """15x45 birlesik maze. Bolumler arasi duvarlar acik."""
    rng = random.Random(seed)
    rows, cols = UNIFIED_ROWS, UNIFIED_COLS
    walls = [
        [{"N": True, "E": True, "S": True, "W": True} for _ in range(cols)]
        for _ in range(rows)
    ]
    vis = [[False] * cols for _ in range(rows)]
    stack = [(0, 0)]
    vis[0][0] = True

    while stack:
        r, c = stack[-1]
        neigh = [
            (d, r + dr, c + dc)
            for d, (dr, dc) in DIRS.items()
            if _in_bounds(r + dr, c + dc, rows, cols)
            and not vis[r + dr][c + dc]
        ]
        if not neigh:
            stack.pop()
            continue
        d, rr, cc = rng.choice(neigh)
        walls[r][c][d] = False
        walls[rr][cc][OPP[d]] = False
        vis[rr][cc] = True
        stack.append((rr, cc))

    walls[0][0]["N"] = False
    walls[rows - 1][cols - 1]["S"] = False

    # Bolum 1-2 ve 2-3 arasi gecisleri ac
    b1 = UNIFIED_SECTION_COLS - 1
    b2 = UNIFIED_SECTION_COLS * 2 - 1
    for r in range(rows):
        walls[r][b1]["E"] = False
        walls[r][b1 + 1]["W"] = False
        walls[r][b2]["E"] = False
        walls[r][b2 + 1]["W"] = False

    walls = open_spawn_area_unified(walls, UNIFIED_SPAWN_CELL, radius=1)
    return walls


def open_spawn_area_unified(walls, spawn_cell, radius=1):
    sr, sc = spawn_cell
    rows, cols = len(walls), len(walls[0])
    for r in range(max(0, sr - radius), min(rows, sr + radius + 1)):
        for c in range(max(0, sc - radius), min(cols, sc + radius + 1)):
            for d, (dr, dc) in DIRS.items():
                nr, nc = r + dr, c + dc
                if _in_bounds(nr, nc, rows, cols):
                    walls[r][c][d] = False
                    walls[nr][nc][OPP[d]] = False
    return walls


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def maze_origin(stage_origin):
    sr, sc = DRONE_SPAWN_CELL
    ox = stage_origin[0] - (sc + 0.5) * CELL_SIZE
    oy = stage_origin[1] - (sr + 0.5) * CELL_SIZE
    return ox, oy


def cell_to_world(r, c, stage_origin):
    ox, oy = maze_origin(stage_origin)
    x = ox + (c + 0.5) * CELL_SIZE
    y = oy + (r + 0.5) * CELL_SIZE
    return x, y


def world_to_cell(x, y, stage_origin, rows=ROWS, cols=COLS):
    """World coords -> (r, c). Use rows/cols for unified maze (15x45)."""
    ox, oy = maze_origin(stage_origin)
    c = int((x - ox) / CELL_SIZE)
    r = int((y - oy) / CELL_SIZE)
    return max(0, min(rows - 1, r)), max(0, min(cols - 1, c))


def world_to_cell_unified(x, y):
    """Unified maze: world coords -> (r, c)."""
    ox, oy = maze_origin_unified()
    c = int((x - ox) / CELL_SIZE)
    r = int((y - oy) / CELL_SIZE)
    return max(0, min(UNIFIED_ROWS - 1, r)), max(0, min(UNIFIED_COLS - 1, c))


# ═══════════════════════════════════════════════════════════════════════════════
# SDF GENERATION & SPAWN
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Segment:
    x: float
    y: float
    length: float
    horizontal: bool


def maze_origin_unified():
    """Birlesik maze icin origin (spawn merkeze gore)."""
    sr, sc = UNIFIED_SPAWN_CELL
    ox = UNIFIED_ORIGIN[0] - (sc + 0.5) * CELL_SIZE
    oy = UNIFIED_ORIGIN[1] - (sr + 0.5) * CELL_SIZE
    return ox, oy


def maze_to_segments(walls, stage_origin):
    return _maze_to_segments_impl(walls, maze_origin(stage_origin), ROWS, COLS)


def maze_to_segments_unified(walls):
    return _maze_to_segments_impl(walls, maze_origin_unified(), UNIFIED_ROWS, UNIFIED_COLS)


def _maze_to_segments_impl(walls, origin_xy, rows, cols):
    ox, oy = origin_xy

    h_edges = [[False] * cols for _ in range(rows + 1)]
    v_edges = [[False] * (cols + 1) for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if walls[r][c]["N"]:
                h_edges[r][c] = True
            if walls[r][c]["S"]:
                h_edges[r + 1][c] = True
            if walls[r][c]["W"]:
                v_edges[r][c] = True
            if walls[r][c]["E"]:
                v_edges[r][c + 1] = True

    segs = []

    for r in range(rows + 1):
        c = 0
        while c < cols:
            if not h_edges[r][c]:
                c += 1
                continue
            start = c
            while c < cols and h_edges[r][c]:
                c += 1
            length = (c - start) * CELL_SIZE
            x = ox + (start + c) * CELL_SIZE / 2.0
            y = oy + r * CELL_SIZE
            segs.append(Segment(x, y, length, True))

    for c in range(cols + 1):
        r = 0
        while r < rows:
            if not v_edges[r][c]:
                r += 1
                continue
            start = r
            while r < rows and v_edges[r][c]:
                r += 1
            length = (r - start) * CELL_SIZE
            x = ox + c * CELL_SIZE
            y = oy + (start + r) * CELL_SIZE / 2.0
            segs.append(Segment(x, y, length, False))

    return segs


def build_maze_sdf(model_name, segments):
    links = []
    for i, seg in enumerate(segments):
        sx = seg.length if seg.horizontal else WALL_THICKNESS
        sy = WALL_THICKNESS if seg.horizontal else seg.length
        links.append(
            f'    <link name="wall_{i}">'
            f'<pose>{seg.x} {seg.y} {Z_CENTER} 0 0 0</pose>'
            f'<collision name="c"><geometry><box><size>{sx} {sy} {WALL_HEIGHT}'
            f'</size></box></geometry></collision>'
            f'<visual name="v"><geometry><box><size>{sx} {sy} {WALL_HEIGHT}'
            f'</size></box></geometry>'
            f'<material><ambient>0.2 0.2 0.2 1</ambient>'
            f'<diffuse>0.2 0.2 0.2 1</diffuse></material></visual></link>'
        )
    sdf = (
        '<?xml version="1.0"?><sdf version="1.9">'
        f'<model name="{model_name}"><static>true</static>'
        f'{"".join(links)}</model></sdf>'
    )
    return sdf


def spawn_sdf_model(model_name, sdf_string):
    path = f"/tmp/{model_name}.sdf"
    with open(path, "w") as f:
        f.write(sdf_string)
    req_str = f'sdf_filename: "{path}"'
    result = subprocess.run(
        [
            "gz", "service", "-s", f"/world/{WORLD_NAME}/create",
            "--reqtype", "gz.msgs.EntityFactory",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "5000",
            "--req", req_str,
        ],
        shell=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"HATA: {model_name} spawn edilemedi: {result.stderr or result.stdout}")
    else:
        print(f"Spawned: {model_name}")


def remove_entity(name, entity_type="MODEL"):
    req_str = f'name: "{name}" type: {entity_type}'
    subprocess.run(
        [
            "gz", "service", "-s", f"/world/{WORLD_NAME}/remove",
            "--reqtype", "gz.msgs.Entity",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "2000",
            "--req", req_str,
        ],
        shell=False,
        capture_output=True,
        text=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: STATIK ENGEL BLOKLARI
# ═══════════════════════════════════════════════════════════════════════════════

def cell_to_world_unified(r, c):
    ox, oy = maze_origin_unified()
    x = ox + (c + 0.5) * CELL_SIZE
    y = oy + (r + 0.5) * CELL_SIZE
    return x, y


def generate_obstacle_positions(walls, stage_origin, ratio=0.25, seed=42):
    rng = random.Random(seed)
    obstacles = []
    sr, sc = DRONE_SPAWN_CELL

    for r in range(ROWS):
        for c in range(COLS):
            if abs(r - sr) <= 2 and abs(c - sc) <= 2:
                continue
            if rng.random() > ratio:
                continue

            open_count = sum(1 for d in ["N", "S", "E", "W"] if not walls[r][c][d])
            if open_count < 2:
                continue

            cx, cy = cell_to_world(r, c, stage_origin)
            side_x = rng.choice([-1.3, 1.3])
            side_y = rng.uniform(-0.8, 0.8)
            obstacles.append((cx + side_x, cy + side_y))

    return obstacles


def generate_obstacle_positions_unified(walls, ratio=0.25, seed=42):
    """Birlesik maze: sadece bolum 2 ve 3'e (c>=15) engel ekle."""
    rng = random.Random(seed)
    obstacles = []
    rows, cols = UNIFIED_ROWS, UNIFIED_COLS

    for r in range(rows):
        for c in range(cols):
            if c < UNIFIED_SECTION_COLS:  # Bolum 1'de engel yok
                continue
            # Bolum 2 merkezi: c = UNIFIED_SECTION_COLS + UNIFIED_SECTION_COLS//2
            section2_center_c = UNIFIED_SECTION_COLS + UNIFIED_SECTION_COLS // 2
            if abs(r - UNIFIED_SPAWN_CELL[0]) <= 2 and abs(c - section2_center_c) <= 2:
                continue
            if rng.random() > ratio:
                continue

            open_count = sum(1 for d in ["N", "S", "E", "W"] if not walls[r][c][d])
            if open_count < 2:
                continue

            cx, cy = cell_to_world_unified(r, c)
            side_x = rng.choice([-1.3, 1.3])
            side_y = rng.uniform(-0.8, 0.8)
            obstacles.append((cx + side_x, cy + side_y))

    return obstacles


def build_obstacles_sdf(model_name, obstacles):
    links = []
    for i, (ox, oy) in enumerate(obstacles):
        links.append(
            f'    <link name="obs_{i}">'
            f'<pose>{ox} {oy} {Z_CENTER} 0 0 0</pose>'
            f'<collision name="c"><geometry><box><size>0.8 0.8 {WALL_HEIGHT}'
            f'</size></box></geometry></collision>'
            f'<visual name="v"><geometry><box><size>0.8 0.8 {WALL_HEIGHT}'
            f'</size></box></geometry>'
            f'<material><ambient>0.5 0.1 0.1 1</ambient>'
            f'<diffuse>0.5 0.1 0.1 1</diffuse></material></visual></link>'
        )
    sdf = (
        '<?xml version="1.0"?><sdf version="1.9">'
        f'<model name="{model_name}"><static>true</static>'
        f'{"".join(links)}</model></sdf>'
    )
    return sdf


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: DINAMIK AKTORLER
# ═══════════════════════════════════════════════════════════════════════════════

def _file_uri(abs_path):
    return "file:///" + os.path.abspath(abs_path).lstrip("/")


def collect_corridors(walls, min_len=2):
    return _collect_corridors_impl(walls, ROWS, COLS, min_len)


def _collect_corridors_impl(walls, rows, cols, min_len=2):
    corridors = []
    for r in range(rows):
        c = 0
        while c < cols:
            if c < cols - 1 and not walls[r][c]["E"]:
                start = c
                while c < cols - 1 and not walls[r][c]["E"]:
                    c += 1
                if c - start + 1 >= min_len:
                    corridors.append(((r, start), (r, c), True))
            c += 1

    for c in range(cols):
        r = 0
        while r < rows:
            if r < rows - 1 and not walls[r][c]["S"]:
                start = r
                while r < rows - 1 and not walls[r][c]["S"]:
                    r += 1
                if r - start + 1 >= min_len:
                    corridors.append(((start, c), (r, c), False))
            r += 1

    return corridors


def collect_corridors_unified(walls, section=3):
    """section=3: sadece bolum 3 (c >= UNIFIED_SECTION_COLS*2) koridorlari."""
    all_c = _collect_corridors_impl(walls, UNIFIED_ROWS, UNIFIED_COLS, min_len=2)
    if section == 3:
        min_c = UNIFIED_SECTION_COLS * 2
        return [c for c in all_c if c[0][1] >= min_c and c[1][1] >= min_c]
    return all_c


def _corridor_near_spawn(corridor, radius, spawn_cell=None):
    (r1, c1), (r2, c2), _ = corridor
    sr, sc = spawn_cell if spawn_cell is not None else DRONE_SPAWN_CELL
    return (abs(r1 - sr) < radius and abs(c1 - sc) < radius) or \
           (abs(r2 - sr) < radius and abs(c2 - sc) < radius)


def corridor_to_world(corridor, stage_origin):
    (r1, c1), (r2, c2), is_horiz = corridor
    x1, y1 = cell_to_world(r1, c1, stage_origin)
    x2, y2 = cell_to_world(r2, c2, stage_origin)
    return x1, y1, x2, y2, is_horiz


def corridor_to_world_unified(corridor):
    (r1, c1), (r2, c2), is_horiz = corridor
    x1, y1 = cell_to_world_unified(r1, c1)
    x2, y2 = cell_to_world_unified(r2, c2)
    return x1, y1, x2, y2, is_horiz


def build_actor_sdf(actor_name, skin_uri, anim_uri, x1, y1, x2, y2, is_horiz,
                    speed=1.4):
    yaw_fwd = 0.0 if is_horiz else 1.57
    yaw_back = 3.14 if is_horiz else -1.57

    dx = x2 - x1
    dy = y2 - y1
    actor_z = 0.9
    dist = math.sqrt(dx * dx + dy * dy)
    duration = dist / speed

    t0 = 0.0
    t1 = duration
    t2 = t1 + 0.5
    t3 = t2 + duration
    t4 = t3 + 0.5

    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <actor name="{actor_name}">
    <pose>{x1} {y1} {actor_z} 0 0 0</pose>
    <skin><filename>{skin_uri}</filename><scale>1.0</scale></skin>
    <animation name="walking"><filename>{anim_uri}</filename>
      <interpolate_x>true</interpolate_x></animation>
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


def _spawn_actors_from_corridors(safe_corridors, world_fn, area, name_prefix, rng_seed=303):
    """Ortak actor spawn mantigi. world_fn(corridor) -> (x1,y1,x2,y2,is_horiz)."""
    if not os.path.exists(SKIN_PATH):
        print(f"Actor dosyalari bulunamadi: {SKIN_PATH}")
        return []

    target_n = max(ACTOR_MIN, min(ACTOR_MAX, int(area * 0.9)))
    actor_slots = []
    for corr in safe_corridors:
        actor_slots.append(corr)
        corr_len = abs(corr[1][0] - corr[0][0]) + abs(corr[1][1] - corr[0][1]) + 1
        if corr_len >= LONG_CORRIDOR_BONUS:
            for _ in range(corr_len // LONG_CORRIDOR_BONUS):
                actor_slots.append(corr)

    rng = random.Random(rng_seed)
    while len(actor_slots) < target_n:
        actor_slots.append(rng.choice(safe_corridors))
    rng.shuffle(actor_slots)
    actor_slots = actor_slots[:target_n]

    skin_uri = _file_uri(SKIN_PATH)
    anim_uri = _file_uri(ANIM_PATH)
    actor_data_list = []

    for idx, corr in enumerate(actor_slots):
        x1, y1, x2, y2, is_horiz = world_fn(corr)
        speed = rng.uniform(ACTOR_SPEED_MIN, ACTOR_SPEED_MAX)
        name = f"{name_prefix}_{idx}_{rng.randint(100, 999)}"
        sdf = build_actor_sdf(name, skin_uri, anim_uri, x1, y1, x2, y2, is_horiz, speed=speed)
        spawn_sdf_model(name, sdf)

        dx, dy = x2 - x1, y2 - y1
        dist = math.sqrt(dx * dx + dy * dy)
        duration = dist / speed
        period = 2.0 * duration + 1.0

        actor_data_list.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "period": round(period, 3),
            "speed": round(speed, 3),
        })
        if (idx + 1) % 20 == 0:
            print(f"  Actors: {idx + 1}/{target_n}")

    return actor_data_list


def spawn_stage3_actors(walls, stage_origin=None):
    if stage_origin is None:
        stage_origin = STAGE_ORIGINS[3]

    corridors = collect_corridors(walls, min_len=2)
    safe_corridors = [
        c for c in corridors if not _corridor_near_spawn(c, AVOID_SPAWN_RADIUS)
    ]
    if not safe_corridors:
        print("Uygun koridor bulunamadi.")
        return []

    world_fn = lambda c: corridor_to_world(c, stage_origin)
    actor_data_list = _spawn_actors_from_corridors(
        safe_corridors, world_fn, ROWS * COLS, "s3_human"
    )
    print(f"Stage 3: {len(actor_data_list)} aktor spawn edildi.")
    return actor_data_list


def spawn_stage3_actors_lazy():
    """Training script tarafindan Stage 3 gecisinde cagrilir."""
    walls = load_walls(WALLS_PATHS[3])
    actor_data = spawn_stage3_actors(walls, STAGE_ORIGINS[3])
    save_actor_data(actor_data, ACTORS_JSON_PATH)
    return actor_data


def spawn_unified_actors(walls):
    """Birlesik maze: sadece bolum 3 (c >= UNIFIED_SECTION_COLS*2) koridorlarina aktor."""
    corridors = collect_corridors_unified(walls, section=3)
    safe_corridors = corridors  # Bolum 3 zaten spawn'dan uzak
    if not safe_corridors:
        print("Bolum 3'te uygun koridor bulunamadi.")
        return []

    actor_data_list = _spawn_actors_from_corridors(
        safe_corridors,
        corridor_to_world_unified,
        UNIFIED_SECTION_COLS * UNIFIED_ROWS,
        "unified_human",
    )
    print(f"Unified Bolum 3: {len(actor_data_list)} aktor spawn edildi.")
    return actor_data_list


def save_actor_data_unified(actor_list):
    data = {
        "spawn_time": time.time(),
        "stage_origin": list(UNIFIED_ORIGIN),
        "unified": True,
        "actors": actor_list,
    }
    with open(ACTORS_UNIFIED_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Actor data kaydedildi: {ACTORS_UNIFIED_JSON_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# FILE I/O
# ═══════════════════════════════════════════════════════════════════════════════

def save_walls(walls, path):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w") as f:
        json.dump(walls, f)
    print(f"Walls kaydedildi: {path}")


def load_walls(path):
    with open(path) as f:
        return json.load(f)


def save_actor_data(actor_list, path):
    data = {
        "spawn_time": time.time(),
        "stage_origin": list(STAGE_ORIGINS[3]),
        "actors": actor_list,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Actor data kaydedildi: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def spawn_all_mazes():
    print("=" * 60)
    print("MEGA-WORLD MAZE CURRICULUM")
    print("=" * 60)

    # --- Stage 1: Basit maze, spawn bolge acik ---
    print("\n[Stage 1] Basit maze (0, 0) ...")
    walls1 = generate_maze(seed=101)
    walls1 = open_spawn_area(walls1, radius=1)
    save_walls(walls1, WALLS_PATHS[1])

    segs1 = maze_to_segments(walls1, STAGE_ORIGINS[1])
    sdf1 = build_maze_sdf("maze_stage1", segs1)
    spawn_sdf_model("maze_stage1", sdf1)

    # --- Stage 2: Karmasik maze + statik engeller ---
    print("\n[Stage 2] Karmasik maze + engeller (1000, 0) ...")
    walls2 = generate_maze(seed=202)
    save_walls(walls2, WALLS_PATHS[2])

    segs2 = maze_to_segments(walls2, STAGE_ORIGINS[2])
    sdf2 = build_maze_sdf("maze_stage2", segs2)
    spawn_sdf_model("maze_stage2", sdf2)

    obs2 = generate_obstacle_positions(walls2, STAGE_ORIGINS[2], ratio=0.25, seed=42)
    if obs2:
        obs_sdf2 = build_obstacles_sdf("obstacles_stage2", obs2)
        spawn_sdf_model("obstacles_stage2", obs_sdf2)
    print(f"  {len(obs2)} statik engel eklendi.")

    # --- Stage 3: Stage 2 ile ayni topoloji + ayni engeller (aktorler lazy) ---
    print("\n[Stage 3] Ayni maze + engeller (2000, 0) ...")
    save_walls(walls2, WALLS_PATHS[3])

    segs3 = maze_to_segments(walls2, STAGE_ORIGINS[3])
    sdf3 = build_maze_sdf("maze_stage3", segs3)
    spawn_sdf_model("maze_stage3", sdf3)

    obs3 = generate_obstacle_positions(walls2, STAGE_ORIGINS[3], ratio=0.25, seed=42)
    if obs3:
        obs_sdf3 = build_obstacles_sdf("obstacles_stage3", obs3)
        spawn_sdf_model("obstacles_stage3", obs_sdf3)

    # --- Aktorler (opsiyonel) ---
    if "--actors" in sys.argv:
        print("\n[Stage 3] Aktorler spawn ediliyor ...")
        actor_data = spawn_stage3_actors(walls2, STAGE_ORIGINS[3])
        save_actor_data(actor_data, ACTORS_JSON_PATH)

    print("\n" + "=" * 60)
    print("MEGA-WORLD HAZIR")
    print(f"  Stage 1: (0, 0)     - Basit maze")
    print(f"  Stage 2: (1000, 0)  - Karmasik + {len(obs2)} engel")
    print(f"  Stage 3: (2000, 0)  - Ayni + aktorler {'SPAWNED' if '--actors' in sys.argv else 'LAZY'}")
    print("=" * 60)


def spawn_unified_maze():
    """Tek birlesik maze: 3 bolum yan yana, fiziksel gecis."""
    print("=" * 60)
    print("BIRLESIK MAZE (Unified)")
    print("=" * 60)

    print("\n[Birlesik] 15x45 maze olusturuluyor...")
    walls = generate_unified_maze(seed=101)
    save_walls(walls, WALLS_UNIFIED_PATH)

    segs = maze_to_segments_unified(walls)
    sdf = build_maze_sdf("maze_unified", segs)
    spawn_sdf_model("maze_unified", sdf)

    obs = generate_obstacle_positions_unified(walls, ratio=0.25, seed=42)
    if obs:
        obs_sdf = build_obstacles_sdf("obstacles_unified", obs)
        spawn_sdf_model("obstacles_unified", obs_sdf)
    print(f"  {len(obs)} statik engel (Bolum 2+3) eklendi.")

    if "--actors" in sys.argv:
        print("\n[Bolum 3] Aktorler spawn ediliyor...")
        actor_data = spawn_unified_actors(walls)
        save_actor_data_unified(actor_data)

    print("\n" + "=" * 60)
    print("BIRLESIK MAZE HAZIR")
    print(f"  Origin: {UNIFIED_ORIGIN} | Spawn: {UNIFIED_SPAWN_CELL}")
    print(f"  Bolum 1: x < {SECTION1_X_MAX} | Bolum 2: {SECTION1_X_MAX}-{SECTION2_X_MAX} | Bolum 3: x >= {SECTION2_X_MAX}")
    print("=" * 60)


def main():
    if "--unified" in sys.argv:
        spawn_unified_maze()
    else:
        spawn_all_mazes()


if __name__ == "__main__":
    main()
