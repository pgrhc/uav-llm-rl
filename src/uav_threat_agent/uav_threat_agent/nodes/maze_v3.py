#!/usr/bin/env python3
import subprocess, time, random, os, math, json, threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header, Int32


WORLD_NAME               = "default"
WALL_HEIGHT              = 7.0
WALL_THICKNESS           = 0.2
CELL_SIZE                = 3.0
Z_CENTER                 = WALL_HEIGHT / 2.0

ROWS, COLS               = 15, 15
random.seed(time.time_ns())

ACTOR_DENSITY            = 0.9
DRONE_SPAWN_CELL         = (ROWS // 2, COLS // 2)
AVOID_SPAWN_RADIUS_CELLS = 1.0

MIN_ACTORS               = 100
MAX_ACTORS               = 200

LONG_CORRIDOR_BONUS_THRESHOLD = 4
MIN_SPAWN_DIST           = 1.5
SPAWN_DELAY_SEC          = 0.05

WALLS_SAVE_PATH          = "/home/ubuntu/Desktop/maze_walls.json"


TOPIC_SET_STAGE   = "/curriculum/set_stage"    
TOPIC_ACTOR_POSES = "/curriculum/actor_poses"  
TOPIC_STAGE_OUT   = "/curriculum/stage"        


STATIC_ACTORS = [
    {"type": "static", "sdf_path": "/home/ubuntu/Desktop/gazebo_custom_models/casual_female/model.sdf"},
    {"type": "static", "sdf_path": "/home/ubuntu/Desktop/gazebo_custom_models/female_visitor/model.sdf"},
    {"type": "static", "sdf_path": "/home/ubuntu/Desktop/gazebo_custom_models/nurse/model.sdf"},
    {"type": "static", "sdf_path": "/home/ubuntu/Desktop/gazebo_custom_models/standing_person/model.sdf"},
]

DYNAMIC_ACTORS = [
    {"type": "dynamic", "sdf_path": "/home/ubuntu/Desktop/gazebo_custom_models/actor_run/model.sdf",       "speed_range": (1.0, 4.0)},
    # {"type": "dynamic", "sdf_path": "/home/ubuntu/Desktop/gazebo_custom_models/walking_person/model.sdf", "speed_range": (1.0, 1.5)},
]

DIRS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
OPP  = {"N": "S", "S": "N", "E": "W", "W": "E"}


def in_bounds(r, c, rows, cols):
    return 0 <= r < rows and 0 <= c < cols

def pretty_xml(elem):
    return ET.tostring(elem, encoding="unicode")

def find_first_child(parent, tag):
    return next((c for c in parent if c.tag == tag), None)

def remove_direct_children(parent, tag):
    for c in [c for c in list(parent) if c.tag == tag]:
        parent.remove(c)

def resolve_resource_path(raw_path, base_dir):
    raw_path = (raw_path or "").strip()
    if not raw_path:
        return raw_path
    if raw_path.startswith(("file://", "http://", "https://")):
        return raw_path
    if raw_path.startswith("/"):
        return "file://" + raw_path
    if raw_path.startswith("model://"):
        rest = raw_path[len("model://"):].split("/", 1)
        return "file://" + os.path.abspath(os.path.join(base_dir, rest[1] if len(rest) == 2 else ""))
    return "file://" + os.path.abspath(os.path.join(base_dir, raw_path))

def fix_resource_paths(root, base_dir):
    tags = {"uri", "filename", "albedo_map", "normal_map", "metalness_map",
            "roughness_map", "emissive_map", "specular_map"}
    for e in root.iter():
        if e.tag in tags and e.text:
            e.text = resolve_resource_path(e.text, base_dir)

def gz_service(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def spawn_sdf_model(name, sdf_str):
    path = f"/tmp/{name}.sdf"
    with open(path, "w") as f:
        f.write(sdf_str)
    cmd = (f"gz service -s /world/{WORLD_NAME}/create "
           f"--reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean "
           f"--timeout 5000 --req 'sdf_filename: \"{path}\"'")
    res = gz_service(cmd)
    ok  = res.returncode == 0 and "data: true" in (res.stdout or "").lower()
    print(f"{'OK' if ok else 'FAIL'} spawn: {name}")
    if not ok and res.stderr:
        print("  stderr:", res.stderr.strip())
    return ok

def remove_entity(name):
    cmd = (f"gz service -s /world/{WORLD_NAME}/remove "
           f"--reqtype gz.msgs.Entity --reptype gz.msgs.Boolean "
           f"--timeout 2000 --req 'name: \"{name}\" type: MODEL'")
    ok = gz_service(cmd).returncode == 0
    print(f"{'Del' if ok else 'DelFail'}: {name}")
    return ok


def generate_perfect_maze(rows, cols):
    walls = [[{"N": True, "E": True, "S": True, "W": True}
              for _ in range(cols)] for _ in range(rows)]
    vis   = [[False] * cols for _ in range(rows)]
    stack = [(0, 0)]
    vis[0][0] = True
    while stack:
        r, c   = stack[-1]
        neigh  = [(d, r+dr, c+dc) for d, (dr, dc) in DIRS.items()
                  if in_bounds(r+dr, c+dc, rows, cols) and not vis[r+dr][c+dc]]
        if not neigh:
            stack.pop()
            continue
        d, rr, cc = random.choice(neigh)
        walls[r][c][d] = walls[rr][cc][OPP[d]] = False
        vis[rr][cc] = True
        stack.append((rr, cc))
    walls[0][0]["N"] = walls[rows-1][cols-1]["S"] = False
    sr, sc = DRONE_SPAWN_CELL
    if in_bounds(sr, sc, rows, cols):
        for d in ("N", "S", "E", "W"):
            walls[sr][sc][d] = False
        for dr, dc, dd in [(-1, 0, "S"), (1, 0, "N"), (0, -1, "E"), (0, 1, "W")]:
            if in_bounds(sr+dr, sc+dc, rows, cols):
                walls[sr+dr][sc+dc][dd] = False
    return walls

def save_walls(walls):
    with open(WALLS_SAVE_PATH, "w") as f:
        json.dump(walls, f)
    print(f"Walls saved: {WALLS_SAVE_PATH}")


@dataclass
class Segment:
    x: float
    y: float
    length: float
    horizontal: bool

def maze_to_segments(walls, rows, cols, cell_size):
    sr, sc = DRONE_SPAWN_CELL
    ox, oy = -(sc + 0.5) * cell_size, -(sr + 0.5) * cell_size
    h = [[False] * cols for _ in range(rows + 1)]
    v = [[False] * (cols + 1) for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if walls[r][c]["N"]: h[r][c]     = True
            if walls[r][c]["S"]: h[r+1][c]   = True
            if walls[r][c]["W"]: v[r][c]     = True
            if walls[r][c]["E"]: v[r][c+1]   = True
    segs = []
    for r in range(rows + 1):
        c = 0
        while c < cols:
            if not h[r][c]: c += 1; continue
            s = c
            while c < cols and h[r][c]: c += 1
            segs.append(Segment(ox + (s+c)*cell_size/2, oy + r*cell_size, (c-s)*cell_size, True))
    for c in range(cols + 1):
        r = 0
        while r < rows:
            if not v[r][c]: r += 1; continue
            s = r
            while r < rows and v[r][c]: r += 1
            segs.append(Segment(ox + c*cell_size, oy + (s+r)*cell_size/2, (r-s)*cell_size, False))
    return segs

def build_maze_sdf(model_name, segments):
    links = []
    for i, seg in enumerate(segments):
        sx, sy = (seg.length, WALL_THICKNESS) if seg.horizontal else (WALL_THICKNESS, seg.length)
        links.append(
            f'<link name="wall_{i}">'
            f'<pose>{seg.x} {seg.y} {Z_CENTER} 0 0 0</pose>'
            f'<collision name="c"><geometry><box><size>{sx} {sy} {WALL_HEIGHT}</size></box></geometry></collision>'
            f'<visual name="v"><geometry><box><size>{sx} {sy} {WALL_HEIGHT}</size></box></geometry>'
            f'<material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material></visual>'
            f'</link>'
        )
    return (f'<?xml version="1.0"?><sdf version="1.9">'
            f'<model name="{model_name}"><static>true</static>{"".join(links)}</model></sdf>')


def collect_corridors(walls, rows, cols, min_len=2):
    cors = []
    for r in range(rows):
        c = 0
        while c < cols:
            if c < cols-1 and not walls[r][c]["E"]:
                s = c
                while c < cols-1 and not walls[r][c]["E"]: c += 1
                if c - s + 1 >= min_len: cors.append(((r, s), (r, c), True))
            c += 1
    for c in range(cols):
        r = 0
        while r < rows:
            if r < rows-1 and not walls[r][c]["S"]:
                s = r
                while r < rows-1 and not walls[r][c]["S"]: r += 1
                if r - s + 1 >= min_len: cors.append(((s, c), (r, c), False))
            r += 1
    return cors

def near_spawn(cor, sr, sc, radius):
    (r1, c1), (r2, c2), _ = cor
    return ((abs(r1-sr) < radius and abs(c1-sc) < radius) or
            (abs(r2-sr) < radius and abs(c2-sc) < radius))

def cor_len(cor):
    (r1, c1), (r2, c2), _ = cor
    return abs(r2-r1) + abs(c2-c1) + 1

def cor_to_world(cor):
    (r1, c1), (r2, c2), is_h = cor
    sr, sc = DRONE_SPAWN_CELL
    ox, oy = -(sc+0.5)*CELL_SIZE, -(sr+0.5)*CELL_SIZE
    return (ox+(c1+0.5)*CELL_SIZE, oy+(r1+0.5)*CELL_SIZE,
            ox+(c2+0.5)*CELL_SIZE, oy+(r2+0.5)*CELL_SIZE, is_h)

def sample_pts(cor, count):
    (r1, c1), (r2, c2), is_h = cor
    sr, sc = DRONE_SPAWN_CELL
    ox, oy = -(sc+0.5)*CELL_SIZE, -(sr+0.5)*CELL_SIZE
    pts = []
    if is_h:
        for cc in random.choices(range(min(c1, c2), max(c1, c2)+1), k=count):
            pts.append((ox + (cc+0.5)*CELL_SIZE, oy + (r1+0.5)*CELL_SIZE))
    else:
        for rr in random.choices(range(min(r1, r2), max(r1, r2)+1), k=count):
            pts.append((ox + (c1+0.5)*CELL_SIZE, oy + (rr+0.5)*CELL_SIZE))
    return pts

def dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def far_enough(x, y, pos, md):
    return all(dist2((x, y), (px, py)) >= md**2 for px, py in pos)

def pick_n():
    return max(MIN_ACTORS, min(MAX_ACTORS, int(ROWS * COLS * ACTOR_DENSITY)))


def add_pose(parent, x, y, z, yaw):
    p = ET.Element("pose")
    p.text = f"{x} {y} {z} 0 0 {yaw}"
    parent.insert(0, p)

def add_script_static(actor, anim):
    s  = ET.SubElement(actor, "script")
    ET.SubElement(s, "loop").text       = "true"
    ET.SubElement(s, "auto_start").text = "true"
    tr = ET.SubElement(s, "trajectory", {"id": "0", "type": anim})
    wp = ET.SubElement(tr, "waypoint")
    ET.SubElement(wp, "time").text = "0.0"
    ET.SubElement(wp, "pose").text = "0 0 0 0 0 0"

def add_script_dynamic(actor, anim, x1, y1, x2, y2, is_h, speed_range):
    yfwd = 0.0  if is_h else 1.57
    ybk  = 3.14 if is_h else -1.57
    dx, dy   = x2-x1, y2-y1
    dist     = max((dx**2 + dy**2)**0.5, 1e-6)
    spd      = random.uniform(*speed_range)
    dur      = max(dist / spd, 0.5)
    n        = max(int(dist / 1.5), 2)
    off      = 0.8
    nx, ny   = -dy/dist, dx/dist
    pat      = random.choice(["straight", "zigzag", "sine"])

    sc = ET.SubElement(actor, "script")
    ET.SubElement(sc, "loop").text       = "true"
    ET.SubElement(sc, "auto_start").text = "true"
    tr = ET.SubElement(sc, "trajectory", {"id": "0", "type": anim})
    t  = 0.0
    for i in range(n + 1):
        ratio  = i / n
        offset = (off * (1 if i % 2 == 0 else -1) * math.sin(ratio * math.pi) if pat == "zigzag"
                  else off * math.sin(ratio * math.pi * 4) if pat == "sine" else 0.0)
        wx = dx * ratio + nx * offset
        wy = dy * ratio + ny * offset
        t  = 0.0 if i == 0 else t + dur / n
        wp = ET.SubElement(tr, "waypoint")
        ET.SubElement(wp, "time").text = f"{t:.2f}"
        ET.SubElement(wp, "pose").text = f"{wx:.3f} {wy:.3f} 0 0 0 {yfwd}"
    for pose_str, dt in [(f"{dx:.3f} {dy:.3f} 0 0 0 {ybk}", 0.5),
                         (f"0 0 0 0 0 {ybk}", dur),
                         (f"0 0 0 0 0 {yfwd}", 0.5)]:
        t += dt
        wp = ET.SubElement(tr, "waypoint")
        ET.SubElement(wp, "time").text = f"{t:.2f}"
        ET.SubElement(wp, "pose").text = pose_str

def build_actor_sdf(name, profile, x1, y1, x2, y2, is_h):
    try:
        root = ET.parse(profile["sdf_path"]).getroot()
    except Exception as e:
        print(f"Parse error {profile['sdf_path']}: {e}")
        return ""
    fix_resource_paths(root, os.path.dirname(profile["sdf_path"]))
    ae = root.find("actor")
    me = root.find("model")
    if ae is not None:
        ae.attrib["name"] = name
        remove_direct_children(ae, "pose")
        remove_direct_children(ae, "script")
        add_pose(ae, x1, y1, 0.9, 0.0)
        anim  = find_first_child(ae, "animation")
        aname = anim.attrib.get("name", "walking") if anim is not None else "walking"
        if profile["type"] == "static":
            add_script_static(ae, aname)
        else:
            add_script_dynamic(ae, aname, x1, y1, x2, y2, is_h,
                               profile.get("speed_range", (1.0, 1.5)))
        return pretty_xml(root)
    if me is not None:
        me.attrib["name"] = name
        remove_direct_children(me, "pose")
        add_pose(me, x1, y1, 0.0, random.uniform(0, 6.28))
        return pretty_xml(root)
    return ""


class CurriculumManager:
    def __init__(self):
        self._spawned: Dict[str, Tuple[float, float, float, str]] = {}
        self._current_stage: int = 0
        self._lock = threading.Lock()
        self._walls = None

    def set_walls(self, walls):
        self._walls = walls

    def positions(self) -> Dict[str, Tuple[float, float, float]]:
        with self._lock:
            return {n: (x, y, z) for n, (x, y, z, _) in self._spawned.items()}

    def current_stage(self) -> int:
        return self._current_stage

    def transition_to(self, stage: int):
        """Train kodundan gelen stage komutunu uygular (ayri thread'de calisir)."""
        if stage == self._current_stage:
            return

        print(f"\n{'='*55}")
        print(f"CURRICULUM STAGE {stage} BASLIYOR")
        print(f"{'='*55}")

        if stage == 1:
            self._remove_all()
            self._spawn(STATIC_ACTORS, label="static")

        elif stage == 2:
            self._remove_by_type("static")
            self._spawn(DYNAMIC_ACTORS, label="dynamic")

        elif stage == 3:
            self._spawn(STATIC_ACTORS, label="static")

        else:
            print(f"Unknown stage: {stage}")
            return

        self._current_stage = stage
        with self._lock:
            n = len(self._spawned)
        print(f"Stage {stage} hazir — aktif aktor: {n}")

    def _remove_all(self):
        with self._lock:
            names = list(self._spawned.keys())
        for n in names:
            remove_entity(n)
            with self._lock:
                self._spawned.pop(n, None)
            time.sleep(0.03)
        print("Tum aktorler silindi.")

    def _remove_by_type(self, actor_type: str):
        with self._lock:
            names = [n for n, (x, y, z, t) in self._spawned.items() if t == actor_type]
        for n in names:
            remove_entity(n)
            with self._lock:
                self._spawned.pop(n, None)
            time.sleep(0.03)
        print(f"{len(names)} '{actor_type}' aktor silindi.")

    def _spawn(self, catalog: List[dict], label: str):
        if self._walls is None:
            print("Walls ayarlanmamis!")
            return

        sr, sc = DRONE_SPAWN_CELL
        cors   = [c for c in collect_corridors(self._walls, ROWS, COLS, min_len=2)
                  if not near_spawn(c, sr, sc, AVOID_SPAWN_RADIUS_CELLS)]
        if not cors:
            print("Guvenli koridor yok")
            return

        n     = pick_n()
        slots = list(cors)
        for cor in cors:
            if cor_len(cor) >= LONG_CORRIDOR_BONUS_THRESHOLD:
                slots.extend([cor] * (cor_len(cor) // LONG_CORRIDOR_BONUS_THRESHOLD))
        while len(slots) < n:
            slots.append(random.choice(cors))
        random.shuffle(slots)
        slots = slots[:n]

        usage     = {}
        for cor in slots:
            usage[cor] = usage.get(cor, 0) + 1
        pts_cache = {cor: sample_pts(cor, cnt) for cor, cnt in usage.items()}

        with self._lock:
            used_pos = [(x, y) for x, y, z, _ in self._spawned.values()]

        per_cor   = {}
        spawned_n = 0
        print(f"Hedef '{label}': {n}")

        for idx, cor in enumerate(slots):
            per_cor[cor] = per_cor.get(cor, 0) + 1
            x1, y1 = pts_cache[cor][per_cor[cor] - 1]

            for _ in range(10):
                if far_enough(x1, y1, used_pos, MIN_SPAWN_DIST):
                    break
                x1, y1 = random.choice(pts_cache[cor])
            if not far_enough(x1, y1, used_pos, MIN_SPAWN_DIST):
                continue

            cx1, cy1, cx2, cy2, is_h = cor_to_world(cor)
            x2, y2 = ((cx1, cy1) if dist2((x1, y1), (cx1, cy1)) > dist2((x1, y1), (cx2, cy2))
                      else (cx2, cy2))

            p    = random.choice(catalog).copy()
            sfx  = "s" if p["type"] == "static" else "d"
            name = f"human_{sfx}_{idx}_{random.randint(100, 999)}"

            sdf = build_actor_sdf(name, p, x1, y1, x2, y2, is_h)
            if not sdf:
                continue
            ok = spawn_sdf_model(name, sdf)
            if ok:
                used_pos.append((x1, y1))
                with self._lock:
                    self._spawned[name] = (x1, y1, 0.0, p["type"])
                spawned_n += 1
            time.sleep(SPAWN_DELAY_SEC)

        print(f"'{label}' {spawned_n} aktor spawn edildi.")


class MazeCurriculumNode(Node):

    def __init__(self, curriculum_manager: CurriculumManager):
        super().__init__("maze_curriculum_node")
        self._mgr = curriculum_manager

        self._sub = self.create_subscription(
            Int32, TOPIC_SET_STAGE, self._on_set_stage, 10)

        self._pose_pub  = self.create_publisher(PoseArray, TOPIC_ACTOR_POSES, 10)
        self._stage_pub = self.create_publisher(Int32,     TOPIC_STAGE_OUT,   10)


        self.create_timer(0.1, self._publish_cb)
        self._transition_thread: Optional[threading.Thread] = None

        self.get_logger().info(
            f"\nMazeCurriculumNode hazir.\n"
            f"  SUB  {TOPIC_SET_STAGE}  (train kodundan stage komutu)\n"
            f"  PUB  {TOPIC_ACTOR_POSES}  (aktör konumlari 10Hz)\n"
            f"  PUB  {TOPIC_STAGE_OUT}        (mevcut stage 10Hz)"
        )

    def _on_set_stage(self, msg: Int32):
        stage = int(msg.data)
        self.get_logger().info(f"set_stage={stage} alindi")

        if self._transition_thread is not None and self._transition_thread.is_alive():
            self.get_logger().warn("Onceki stage gecisi devam ediyor, atlanıyor.")
            return

        self._transition_thread = threading.Thread(
            target=self._mgr.transition_to,
            args=(stage,),
            daemon=True
        )
        self._transition_thread.start()

    def _publish_cb(self):
        positions = self._mgr.positions()
        stage_num = self._mgr.current_stage()

        pa = PoseArray()
        pa.header = Header()
        pa.header.stamp    = self.get_clock().now().to_msg()
        pa.header.frame_id = "world"
        for x, y, z in positions.values():
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = float(z)
            pa.poses.append(p)
        self._pose_pub.publish(pa)

        si = Int32()
        si.data = stage_num
        self._stage_pub.publish(si)

def main():
    print("Maze olusturuluyor...")
    walls = generate_perfect_maze(ROWS, COLS)
    save_walls(walls)
    segs = maze_to_segments(walls, ROWS, COLS, CELL_SIZE)
    spawn_sdf_model("maze_current", build_maze_sdf("maze_current", segs))
    time.sleep(1.5)

    mgr = CurriculumManager()
    mgr.set_walls(walls)

    rclpy.init()
    node = MazeCurriculumNode(mgr)

    print(
        "\nROS2 node calisiyor. Train basladiginda stage komutlarini bekliyor...\n"
        f"  SUB  {TOPIC_SET_STAGE}   <- CurriculumScheduler publish eder\n"
        f"  PUB  {TOPIC_ACTOR_POSES}  -> aktor konumlari\n"
        f"  PUB  {TOPIC_STAGE_OUT}        -> mevcut stage\n"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Durduruldu.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()