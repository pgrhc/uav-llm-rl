import subprocess
import time
import random
import os
import math
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────────────────────
WORLD_NAME = "default"
WALL_HEIGHT = 7.0
WALL_THICKNESS = 0.2
CELL_SIZE = 3.0
Z_CENTER = WALL_HEIGHT / 2.0

ROWS, COLS = 15, 15
random.seed(time.time_ns())

ACTOR_DENSITY = 0.9
DRONE_SPAWN_CELL = (ROWS // 2, COLS // 2)
AVOID_SPAWN_RADIUS_CELLS = 1.0

# Stabil deneme için önce küçük tut
MIN_ACTORS = 20
MAX_ACTORS = 20

LONG_CORRIDOR_BONUS_THRESHOLD = 4
MIN_SPAWN_DIST = 1.5
SPAWN_DELAY_SEC = 0.05

WALLS_SAVE_PATH = "/home/ubuntu/Desktop/maze_walls.json"

ACTOR_CATALOG = [
    {
        "type": "dynamic",
        "sdf_path": "/home/ubuntu/Desktop/gazebo_custom_models/actor_run/model.sdf",
        "speed_range": (2.5, 4.0)
    },
    {
        "type": "static",
        "sdf_path": "/home/ubuntu/Desktop/gazebo_custom_models/casual_female/model.sdf",
    },
    {
        "type": "static",
        "sdf_path": "/home/ubuntu/Desktop/gazebo_custom_models/female_visitor/model.sdf",
    },
    {
        "type": "static",
        "sdf_path": "/home/ubuntu/Desktop/gazebo_custom_models/nurse/model.sdf",
    },
    # {
    #     "type": "dynamic",
    #     "sdf_path": "/home/ubuntu/Desktop/gazebo_custom_models/male_visitor/model.sdf",
    #     "speed_range": (1.0, 1.5)
    # },
    {
        "type": "static",
        "sdf_path": "/home/ubuntu/Desktop/gazebo_custom_models/standing_person/model.sdf",
    },
    {
        "type": "dynamic",
        "sdf_path": "/home/ubuntu/Desktop/gazebo_custom_models/walking_person/model.sdf",
        "speed_range": (1.0, 1.5)
    },
    # {
    #     "type": "dynamic",
    #     "sdf_path": "/home/ubuntu/Desktop/gazebo_custom_models/actor_multiple_paths/model.sdf",
    #     "speed_range": (1.0, 4.0),
    #     "is_hybrid": True
    # }
]

HYBRID_ANIMS_DIR = "/home/ubuntu/Desktop/gazebo_custom_models/actor_multiple_paths/meshes"
HYBRID_ANIMS = [
    "walk.dae", "run.dae", "moonwalk.dae", "stand.dae",
    "sit.dae", "talk_a.dae", "talk_b.dae"
]

DIRS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}


# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI
# ─────────────────────────────────────────────────────────────────────────────
def in_bounds(r, c, rows, cols):
    return 0 <= r < rows and 0 <= c < cols


def to_file_uri(path: str) -> str:
    path = os.path.abspath(path)
    return "file://" + path


def pretty_xml(elem: ET.Element) -> str:
    return ET.tostring(elem, encoding="unicode")


def find_first_child(parent: ET.Element, tag_name: str):
    for child in parent:
        if child.tag == tag_name:
            return child
    return None


def remove_direct_children(parent: ET.Element, tag_name: str):
    to_remove = [c for c in list(parent) if c.tag == tag_name]
    for c in to_remove:
        parent.remove(c)


def resolve_resource_path(raw_path: str, base_dir: str) -> str:
    raw_path = (raw_path or "").strip()
    if not raw_path:
        return raw_path

    if raw_path.startswith(("file://", "http://", "https://")):
        return raw_path

    if raw_path.startswith("/"):
        return "file://" + raw_path

    if raw_path.startswith("model://"):
        rest = raw_path[len("model://"):]
        parts = rest.split("/", 1)
        rest_rel = parts[1] if len(parts) == 2 else ""
        abs_path = os.path.join(base_dir, rest_rel)
        return "file://" + os.path.abspath(abs_path)

    abs_path = os.path.join(base_dir, raw_path)
    return "file://" + os.path.abspath(abs_path)


def fix_resource_paths_in_tree(root: ET.Element, base_dir: str):
    resource_tags = {
        "uri", "filename",
        "albedo_map", "normal_map", "metalness_map",
        "roughness_map", "emissive_map", "specular_map"
    }

    for elem in root.iter():
        if elem.tag in resource_tags and elem.text:
            elem.text = resolve_resource_path(elem.text, base_dir)


def service_call(cmd: str):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def spawn_sdf_model(model_name: str, sdf_string: str) -> bool:
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

    res = service_call(cmd)

    ok = (res.returncode == 0) and ("data: true" in (res.stdout or "").lower())
    if not ok:
        print(f"❌ Spawn başarısız: {model_name}")
        if res.stdout:
            print("stdout:", res.stdout.strip())
        if res.stderr:
            print("stderr:", res.stderr.strip())
    else:
        print(f"✅ Spawn: {model_name}")

    return ok


def remove_entity(name: str, entity_type: str = "MODEL"):
    cmd = (
        f"gz service -s /world/{WORLD_NAME}/remove "
        f"--reqtype gz.msgs.Entity "
        f"--reptype gz.msgs.Boolean "
        f"--timeout 2000 "
        f"--req 'name: \"{name}\" type: {entity_type}'"
    )
    res = service_call(cmd)
    return res.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# MAZE
# ─────────────────────────────────────────────────────────────────────────────
def generate_perfect_maze(rows, cols):
    walls = [[{"N": True, "E": True, "S": True, "W": True} for _ in range(cols)] for _ in range(rows)]
    vis = [[False] * cols for _ in range(rows)]
    stack = [(0, 0)]
    vis[0][0] = True

    while stack:
        r, c = stack[-1]
        neigh = []
        for d, (dr, dc) in DIRS.items():
            rr, cc = r + dr, c + dc
            if in_bounds(rr, cc, rows, cols) and not vis[rr][cc]:
                neigh.append((d, rr, cc))

        if not neigh:
            stack.pop()
            continue

        d, rr, cc = random.choice(neigh)
        walls[r][c][d] = False
        walls[rr][cc][OPP[d]] = False
        vis[rr][cc] = True
        stack.append((rr, cc))

    walls[0][0]["N"] = False
    walls[rows - 1][cols - 1]["S"] = False

    sr, sc = DRONE_SPAWN_CELL
    if in_bounds(sr, sc, rows, cols):
        walls[sr][sc]["N"] = False
        walls[sr][sc]["S"] = False
        walls[sr][sc]["E"] = False
        walls[sr][sc]["W"] = False

        if in_bounds(sr - 1, sc, rows, cols):
            walls[sr - 1][sc]["S"] = False
        if in_bounds(sr + 1, sc, rows, cols):
            walls[sr + 1][sc]["N"] = False
        if in_bounds(sr, sc - 1, rows, cols):
            walls[sr][sc - 1]["E"] = False
        if in_bounds(sr, sc + 1, rows, cols):
            walls[sr][sc + 1]["W"] = False

    return walls


@dataclass
class Segment:
    x: float
    y: float
    length: float
    horizontal: bool


def maze_to_segments(walls, rows, cols, cell_size):
    spawn_r, spawn_c = DRONE_SPAWN_CELL
    ox = -(spawn_c + 0.5) * cell_size
    oy = -(spawn_r + 0.5) * cell_size

    h = [[False] * cols for _ in range(rows + 1)]
    v = [[False] * (cols + 1) for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if walls[r][c]["N"]:
                h[r][c] = True
            if walls[r][c]["S"]:
                h[r + 1][c] = True
            if walls[r][c]["W"]:
                v[r][c] = True
            if walls[r][c]["E"]:
                v[r][c + 1] = True

    segs = []

    for r in range(rows + 1):
        c = 0
        while c < cols:
            if not h[r][c]:
                c += 1
                continue
            start = c
            while c < cols and h[r][c]:
                c += 1
            length = (c - start) * cell_size
            x = ox + (start + c) * cell_size / 2.0
            y = oy + r * cell_size
            segs.append(Segment(x, y, length, True))

    for c in range(cols + 1):
        r = 0
        while r < rows:
            if not v[r][c]:
                r += 1
                continue
            start = r
            while r < rows and v[r][c]:
                r += 1
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


def collect_corridors(walls, rows, cols, min_len=2):
    corridors = []

    for r in range(rows):
        c = 0
        while c < cols:
            if c < cols - 1 and not walls[r][c]["E"]:
                start = c
                while c < cols - 1 and not walls[r][c]["E"]:
                    c += 1
                length = c - start + 1
                if length >= min_len:
                    corridors.append(((r, start), (r, c), True))
            c += 1

    for c in range(cols):
        r = 0
        while r < rows:
            if r < rows - 1 and not walls[r][c]["S"]:
                start = r
                while r < rows - 1 and not walls[r][c]["S"]:
                    r += 1
                length = r - start + 1
                if length >= min_len:
                    corridors.append(((start, c), (r, c), False))
            r += 1

    return corridors


def corridor_to_world(corridor, rows, cols, cell_size):
    (r1, c1), (r2, c2), is_horiz = corridor
    spawn_r, spawn_c = DRONE_SPAWN_CELL
    ox = -(spawn_c + 0.5) * cell_size
    oy = -(spawn_r + 0.5) * cell_size

    x1 = ox + (c1 + 0.5) * cell_size
    y1 = oy + (r1 + 0.5) * cell_size
    x2 = ox + (c2 + 0.5) * cell_size
    y2 = oy + (r2 + 0.5) * cell_size
    return x1, y1, x2, y2, is_horiz


def corridor_len_cells(corridor):
    (r1, c1), (r2, c2), _ = corridor
    return abs(r2 - r1) + abs(c2 - c1) + 1


def sample_points_on_corridor(corridor, rows, cols, cell_size, count):
    (r1, c1), (r2, c2), is_horiz = corridor
    spawn_r, spawn_c = DRONE_SPAWN_CELL
    ox = -(spawn_c + 0.5) * cell_size
    oy = -(spawn_r + 0.5) * cell_size

    points = []

    if is_horiz:
        cols_list = list(range(min(c1, c2), max(c1, c2) + 1))
        sampled = random.choices(cols_list, k=count)
        for cc in sampled:
            x = ox + (cc + 0.5) * cell_size
            y = oy + (r1 + 0.5) * cell_size
            points.append((x, y))
    else:
        rows_list = list(range(min(r1, r2), max(r1, r2) + 1))
        sampled = random.choices(rows_list, k=count)
        for rr in sampled:
            x = ox + (c1 + 0.5) * cell_size
            y = oy + (rr + 0.5) * cell_size
            points.append((x, y))

    return points


def pick_actor_count(rows, cols):
    area = rows * cols
    n = int(area * ACTOR_DENSITY)
    if n < MIN_ACTORS:
        n = MIN_ACTORS
    if n > MAX_ACTORS:
        n = MAX_ACTORS
    return n


def save_walls(walls, path=WALLS_SAVE_PATH):
    with open(path, "w") as f:
        json.dump(walls, f)
    print(f"✅ Walls kaydedildi: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ACTOR / MODEL SDF ÜRETİMİ
# ─────────────────────────────────────────────────────────────────────────────
def make_pose_text(x, y, z, yaw=0.0):
    return f"{x} {y} {z} 0 0 {yaw}"


def add_pose(parent: ET.Element, x, y, z, yaw):
    pose = ET.Element("pose")
    pose.text = make_pose_text(x, y, z, yaw)
    parent.insert(0, pose)


def add_script_static(actor_elem: ET.Element, anim_name: str):
    script = ET.Element("script")

    loop = ET.SubElement(script, "loop")
    loop.text = "true"

    auto_start = ET.SubElement(script, "auto_start")
    auto_start.text = "true"

    traj = ET.SubElement(script, "trajectory", attrib={"id": "0", "type": anim_name})
    wp = ET.SubElement(traj, "waypoint")
    t = ET.SubElement(wp, "time")
    t.text = "0.0"
    pose = ET.SubElement(wp, "pose")
    pose.text = "0 0 0 0 0 0"

    actor_elem.append(script)


def add_script_dynamic(actor_elem: ET.Element, anim_name: str, x1, y1, x2, y2, is_horiz, speed_range):
    yaw_fwd = 0.0 if is_horiz else 1.57
    yaw_back = 3.14 if is_horiz else -1.57

    dx = x2 - x1
    dy = y2 - y1
    dist = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)

    speed = random.uniform(*speed_range)
    duration = max(dist / speed, 0.5)

    pattern = random.choice(["straight", "zigzag", "sine"])
    num_points = max(int(dist / 1.5), 2)
    max_offset = 0.8

    nx = -dy / dist
    ny = dx / dist

    script = ET.Element("script")

    loop = ET.SubElement(script, "loop")
    loop.text = "true"

    auto_start = ET.SubElement(script, "auto_start")
    auto_start.text = "true"

    traj = ET.SubElement(script, "trajectory", attrib={"id": "0", "type": anim_name})

    t_current = 0.0
    for i in range(num_points + 1):
        t_ratio = i / num_points
        base_x = dx * t_ratio
        base_y = dy * t_ratio

        offset = 0.0
        if pattern == "zigzag":
            offset = max_offset * (1 if i % 2 == 0 else -1) * math.sin(t_ratio * math.pi)
        elif pattern == "sine":
            offset = max_offset * math.sin(t_ratio * math.pi * 4)

        wx = base_x + nx * offset
        wy = base_y + ny * offset

        t_val = 0.0 if i == 0 else t_current + (duration * (1.0 / num_points))
        t_current = t_val

        wp = ET.SubElement(traj, "waypoint")
        t = ET.SubElement(wp, "time")
        t.text = f"{t_val:.2f}"
        pose = ET.SubElement(wp, "pose")
        pose.text = f"{wx:.3f} {wy:.3f} 0 0 0 {yaw_fwd}"

    t_current += 0.5
    wp = ET.SubElement(traj, "waypoint")
    t = ET.SubElement(wp, "time")
    t.text = f"{t_current:.2f}"
    pose = ET.SubElement(wp, "pose")
    pose.text = f"{dx:.3f} {dy:.3f} 0 0 0 {yaw_back}"

    t_current += duration
    wp = ET.SubElement(traj, "waypoint")
    t = ET.SubElement(wp, "time")
    t.text = f"{t_current:.2f}"
    pose = ET.SubElement(wp, "pose")
    pose.text = f"0 0 0 0 0 {yaw_back}"

    t_current += 0.5
    wp = ET.SubElement(traj, "waypoint")
    t = ET.SubElement(wp, "time")
    t.text = f"{t_current:.2f}"
    pose = ET.SubElement(wp, "pose")
    pose.text = f"0 0 0 0 0 {yaw_fwd}"

    actor_elem.append(script)


def build_actor_from_sdf_template(entity_name, profile, x1, y1, x2, y2, is_horiz):
    sdf_path = profile["sdf_path"]

    try:
        tree = ET.parse(sdf_path)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ XML parse hatası: {sdf_path} -> {e}")
        return ""

    base_dir = os.path.dirname(sdf_path)
    fix_resource_paths_in_tree(root, base_dir)

    actor_elem = root.find("actor")
    model_elem = root.find("model")

    # HYBRID ayarı
    chosen_anim = None
    if profile.get("is_hybrid", False) and actor_elem is not None:
        chosen_anim = random.choice(HYBRID_ANIMS)
        anim_uri = "file://" + os.path.join(HYBRID_ANIMS_DIR, chosen_anim)
        unique_anim_name = f"hybrid_anim_{entity_name}"

        first_anim = find_first_child(actor_elem, "animation")
        if first_anim is None:
            first_anim = ET.Element("animation")
            actor_elem.append(first_anim)

        first_anim.attrib["name"] = unique_anim_name

        filename_elem = find_first_child(first_anim, "filename")
        if filename_elem is None:
            filename_elem = ET.SubElement(first_anim, "filename")
        filename_elem.text = anim_uri

        if chosen_anim in ["stand.dae", "sit.dae", "sitting.dae", "talk_a.dae", "talk_b.dae"]:
            profile["type"] = "static"
        else:
            profile["type"] = "dynamic"

    actor_type = profile.get("type", "static")

    # ACTOR
    if actor_elem is not None:
        actor_elem.attrib["name"] = entity_name

        # Sadece root actor altındaki pose/script silinsin
        remove_direct_children(actor_elem, "pose")
        remove_direct_children(actor_elem, "script")

        ACTOR_Z = 0.9
        add_pose(actor_elem, x1, y1, ACTOR_Z, 0.0)

        anim_elem = find_first_child(actor_elem, "animation")
        anim_name = anim_elem.attrib.get("name", "walking") if anim_elem is not None else "walking"

        if actor_type == "static":
            add_script_static(actor_elem, anim_name)
        else:
            speed_range = profile.get("speed_range", (1.0, 1.5))
            add_script_dynamic(actor_elem, anim_name, x1, y1, x2, y2, is_horiz, speed_range)

        return pretty_xml(root)

    # MODEL
    if model_elem is not None:
        model_elem.attrib["name"] = entity_name
        remove_direct_children(model_elem, "pose")

        MODEL_Z = 0.0
        random_yaw = random.uniform(0, 6.28)
        add_pose(model_elem, x1, y1, MODEL_Z, random_yaw)

        return pretty_xml(root)

    print(f"⚠️ Ne actor ne model bulundu: {sdf_path}")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# SPAWN
# ─────────────────────────────────────────────────────────────────────────────
def _corridor_near_spawn(corridor, spawn_r, spawn_c, radius):
    (r1, c1), (r2, c2), _ = corridor
    if (abs(r1 - spawn_r) < radius and abs(c1 - spawn_c) < radius) or \
       (abs(r2 - spawn_r) < radius and abs(c2 - spawn_c) < radius):
        return True
    return False


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def is_far_enough(x, y, positions, min_dist):
    min_d2 = min_dist ** 2
    for px, py in positions:
        if dist2((x, y), (px, py)) < min_d2:
            return False
    return True


def spawn_multiple_actors(walls, rows, cols):
    corridors = collect_corridors(walls, rows, cols, min_len=2)

    sr, sc = DRONE_SPAWN_CELL
    safe_corridors = [
        c for c in corridors
        if not _corridor_near_spawn(c, sr, sc, AVOID_SPAWN_RADIUS_CELLS)
    ]

    if not safe_corridors:
        print("❌ Uygun koridor bulunamadı.")
        return

    target_n = pick_actor_count(rows, cols)

    actor_slots = []
    for corr in safe_corridors:
        actor_slots.append(corr)
        corr_len = corridor_len_cells(corr)
        if corr_len >= LONG_CORRIDOR_BONUS_THRESHOLD:
            bonus = corr_len // LONG_CORRIDOR_BONUS_THRESHOLD
            for _ in range(bonus):
                actor_slots.append(corr)

    while len(actor_slots) < target_n:
        actor_slots.append(random.choice(safe_corridors))

    random.shuffle(actor_slots)
    actor_slots = actor_slots[:target_n]

    print(f"🎯 Hedeflenen İnsan Sayısı: {target_n}")
    print(f"📌 Güvenli koridor sayısı: {len(safe_corridors)}")

    corridor_usage = {}
    for corr in actor_slots:
        corridor_usage[corr] = corridor_usage.get(corr, 0) + 1

    corridor_points_cache = {}
    for corr, count in corridor_usage.items():
        corridor_points_cache[corr] = sample_points_on_corridor(corr, rows, cols, CELL_SIZE, count)

    spawned_per_corridor = {}
    used_positions = []

    for idx, corr in enumerate(actor_slots):
        spawned_per_corridor[corr] = spawned_per_corridor.get(corr, 0) + 1

        points = corridor_points_cache[corr]
        point_idx = spawned_per_corridor[corr] - 1
        x1, y1 = points[point_idx]

        # Çok yakınsa birkaç kez yeni nokta dene
        attempts = 0
        while attempts < 10 and not is_far_enough(x1, y1, used_positions, MIN_SPAWN_DIST):
            x1, y1 = random.choice(points)
            attempts += 1

        if not is_far_enough(x1, y1, used_positions, MIN_SPAWN_DIST):
            print(f"⚠️ {idx} için yeterince boş nokta bulunamadı, geçiliyor.")
            continue

        cx1, cy1, cx2, cy2, is_horiz = corridor_to_world(corr, rows, cols, CELL_SIZE)
        endpoints = [(cx1, cy1), (cx2, cy2)]

        d0 = dist2((x1, y1), endpoints[0])
        d1 = dist2((x1, y1), endpoints[1])
        x2, y2 = endpoints[0] if d0 > d1 else endpoints[1]

        name = f"human_{idx}_{random.randint(100, 999)}"
        chosen_profile = random.choice(ACTOR_CATALOG)

        sdf = build_actor_from_sdf_template(name, chosen_profile.copy(), x1, y1, x2, y2, is_horiz)
        if not sdf:
            print(f"⚠️ SDF üretilemedi: {name}")
            continue

        ok = spawn_sdf_model(name, sdf)
        if ok:
            used_positions.append((x1, y1))

        time.sleep(SPAWN_DELAY_SEC)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🧩 Maze oluşturuluyor...")

    walls = generate_perfect_maze(ROWS, COLS)
    save_walls(walls)

    segments = maze_to_segments(walls, ROWS, COLS, CELL_SIZE)
    maze_sdf = build_maze_sdf("maze_current", segments)
    spawn_sdf_model("maze_current", maze_sdf)

    time.sleep(1.5)
    spawn_multiple_actors(walls, ROWS, COLS)