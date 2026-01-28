import subprocess, time, random
from dataclasses import dataclass
import sys

# --- ROS 2 IMPORTS ---
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped

WORLD_NAME = "default"
WALL_HEIGHT = 7.0
WALL_THICKNESS = 0.2
CELL_SIZE = 5.0
Z_CENTER = WALL_HEIGHT / 2

ROWS, COLS = 25, 25
random.seed(time.time_ns())

# ---- Maze generation (Aynı kalıyor) ----
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

    # Giriş (Sol Üst) ve Çıkış (Sağ Alt) açıklıkları
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
    print(f"✅ Spawned model: {model_name}")

def remove_model(model_name: str):
    cmd = (
        f"gz service -s /world/{WORLD_NAME}/remove "
        f"--reqtype gz.msgs.Entity "
        f"--reptype gz.msgs.Boolean "
        f"--timeout 2000 "
        f"--req 'name: \"{model_name}\" type: MODEL'"
    )
    subprocess.run(cmd, shell=True, capture_output=True, text=True)


# ---- YENİ: ÇIKIŞ HESAPLAMA VE YAYINLAMA ----
def publish_exit_pose(node):
    # 1. Çıkış hücresini bul (Sağ Alt Köşe)
    exit_row = ROWS - 1
    exit_col = COLS - 1
    
    # 2. Dünya koordinatlarını hesapla (maze_to_segments mantığıyla aynı)
    width, height = COLS * CELL_SIZE, ROWS * CELL_SIZE
    ox, oy = -width / 2, -height / 2
    
    # Hücrenin tam merkezini hedef alıyoruz
    target_x = ox + (exit_col * CELL_SIZE) + (CELL_SIZE / 2)
    target_y = oy + (exit_row * CELL_SIZE) + (CELL_SIZE / 2)
    
    print(f"📍 Calculated Exit Coordinate: X={target_x:.2f}, Y={target_y:.2f}")

    # 3. ROS Mesajını Hazırla
    msg = PoseStamped()
    msg.header.frame_id = 'map'  # Nav2 'map' frame'ini kullanır
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.position.x = target_x
    msg.pose.position.y = target_y
    msg.pose.orientation.w = 1.0 # Dönüş açısı önemli değil

    # 4. Yayıncıyı oluştur (Transient Local = Sonradan gelenler de görsün)
    qos = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE
    )
    publisher = node.create_publisher(PoseStamped, '/maze/exit', qos)
    
    # Mesajı yayınla
    publisher.publish(msg)
    print(f"📡 Published exit pose to topic '/maze/exit'")
    return target_x, target_y


def main():
    # Önce ROS'u başlat
    rclpy.init()
    node = rclpy.create_node('maze_spawner_node')

    # Labirenti oluştur ve spawn et
    model_name = "maze_current"
    print(f"🧩 Generating perfect maze: {ROWS}x{COLS}")
    remove_model(model_name)
    time.sleep(0.5) # Eskisinin silinmesi için bekle
    
    walls = generate_perfect_maze(ROWS, COLS)
    segments = maze_to_segments(walls, ROWS, COLS, CELL_SIZE)
    sdf = build_single_model_sdf(model_name, segments)
    spawn_sdf_model(model_name, sdf)

    # Çıkış noktasını yayınla
    publish_exit_pose(node)

    print("ℹ️  Node is running to keep the topic alive. Press Ctrl+C to stop.")
    try:
        # Node'u açık tutuyoruz ki 'Transient Local' mesaj hafızada kalsın
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()