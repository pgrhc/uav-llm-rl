import subprocess, time, random
from dataclasses import dataclass
import time
import subprocess

WORLD_NAME = "default"
WALL_HEIGHT = 7.0
WALL_THICKNESS = 0.2
CELL_SIZE = 5.0
Z_CENTER = WALL_HEIGHT / 2

ROWS, COLS = 25, 25
random.seed(time.time_ns())

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
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    print(f"✅ Spawned model: {model_name}")

# ---- Maze generation (same as before) ----
DIRS = {"N": (-1,0), "E": (0,1), "S": (1,0), "W": (0,-1)}
OPP  = {"N":"S","S":"N","E":"W","W":"E"}

def in_bounds(r,c,rows,cols): return 0 <= r < rows and 0 <= c < cols

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

    walls[0][0]["N"] = False
    walls[rows-1][cols-1]["S"] = False
    return walls

@dataclass
class Segment:
    x: float
    y: float
    length: float
    horizontal: bool

def maze_to_segments(walls, rows, cols, cell_size):
    width, height = cols*cell_size, rows*cell_size
    ox, oy = -width/2, -height/2

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
            x = ox + (start*cell_size + end*cell_size)/2
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
            y = oy + (start*cell_size + end*cell_size)/2
            segs.append(Segment(x,y,length,False))

    return segs

def build_single_model_sdf(model_name: str, segments):
    links = []
    for i, seg in enumerate(segments):
        if seg.horizontal:
            sx, sy = seg.length, WALL_THICKNESS
        else:
            sx, sy = WALL_THICKNESS, seg.length

        link = f"""
    <link name="wall_{i}">
      <pose>{seg.x} {seg.y} {Z_CENTER} 0 0 0</pose>
      <collision name="col">
        <geometry><box><size>{sx} {sy} {WALL_HEIGHT}</size></box></geometry>
      </collision>
      <visual name="vis">
        <geometry><box><size>{sx} {sy} {WALL_HEIGHT}</size></box></geometry>
        <material>
          <ambient>0 0 0 1</ambient>
          <diffuse>0 0 0 1</diffuse>
        </material>
      </visual>
    </link>
"""
        links.append(link)

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{model_name}">
    <static>true</static>
    {''.join(links)}
  </model>
</sdf>
"""
    return sdf

def remove_model(model_name: str):
    # Gazebo'da model varsa siler, yoksa hata verse bile biz devam ederiz
    cmd = (
        f"gz service -s /world/{WORLD_NAME}/remove "
        f"--reqtype gz.msgs.Entity "
        f"--reptype gz.msgs.Boolean "
        f"--timeout 2000 "
        f"--req 'name: \"{model_name}\" type: MODEL'"
    )
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    # Bazı sürümlerde model yoksa false dönebilir; bunu fatal saymıyoruz
    return res.returncode == 0

def spawn_maze():
    model_name = "maze_current"
    print(f"🧩 Generating perfect maze: {ROWS}x{COLS} (random, solvable)")
    print("🧹 Removing old maze (if any)...")
    remove_model(model_name)
    time.sleep(0.2)
    walls = generate_perfect_maze(ROWS, COLS)
    segments = maze_to_segments(walls, ROWS, COLS, CELL_SIZE)
    print(f"🧱 Built {len(segments)} wall segments (merged)")
    sdf = build_single_model_sdf(model_name, segments)
    spawn_sdf_model(model_name, sdf)

if __name__ == "__main__":
    spawn_maze()