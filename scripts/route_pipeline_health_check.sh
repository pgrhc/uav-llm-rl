#!/usr/bin/env bash
# =============================================================================
# ROUTE CURRICULUM — uçtan uca sağlık kontrolü (tanı raporu Bölüm 1 ile uyumlu)
# Tüm çıktıyı hem terminale basar hem .txt dosyasına yazar.
#
# Kullanım:
#   ./scripts/route_pipeline_health_check.sh
#   ./scripts/route_pipeline_health_check.sh /path/to/rapor.txt
#
# GIL / eğitim yükü (3 topic paralel hz — tüm scripti tekrar çalıştırmadan):
#   ./scripts/route_pipeline_health_check.sh gil-check
#   ./scripts/route_pipeline_health_check.sh gil-check /path/to/aynı_log.txt   # baseline + train için append
#
# Ortam (isteğe bağlı):
#   export ROUTE_HEALTH_WS_SETUP=/home/ubuntu/Desktop/ros2_env/uav_ws/install/setup.bash
#   export HZ_SAMPLE_SEC=12          # tam rapor: topic başına örnekleme (saniye)
#   export GIL_CHECK_SEC=20          # gil-check: paralel ölçüm süresi
#   export GIL_CHECK_WINDOW=30      # gil-check: ros2 topic hz -w
#
# Not: ros2 topic hz sonsuz döner; timeout ile kesilir. Sim + node'lar açıkken çalıştırın.
# =============================================================================

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HZ_SAMPLE_SEC="${HZ_SAMPLE_SEC:-12}"
GIL_CHECK_SEC="${GIL_CHECK_SEC:-20}"
GIL_CHECK_WINDOW="${GIL_CHECK_WINDOW:-30}"

MODE="full"
if [[ "${1:-}" == "gil-check" || "${1:-}" == "--gil-check" ]]; then
  MODE="gil-check"
  shift
fi

if [[ "$MODE" == "gil-check" ]]; then
  LOG_FILE="${1:-$REPO_ROOT/logs/route_gil_snapshot_$(date +%Y%m%d_%H%M%S).txt}"
  [[ -n "${1:-}" ]] && shift
else
  LOG_FILE="${1:-$REPO_ROOT/logs/route_pipeline_health_$(date +%Y%m%d_%H%M%S).txt}"
  [[ -n "${1:-}" ]] && shift
fi
mkdir -p "$(dirname "$LOG_FILE")"

source_ros_env() {
  if [[ -n "${ROUTE_HEALTH_WS_SETUP:-}" && -f "$ROUTE_HEALTH_WS_SETUP" ]]; then
    # shellcheck source=/dev/null
    source "$ROUTE_HEALTH_WS_SETUP"
  elif [[ -f "$HOME/Desktop/ros2_env/install/setup.bash" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/Desktop/ros2_env/install/setup.bash"
  elif [[ -f "$HOME/Desktop/ros2_env/uav_ws/install/setup.bash" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/Desktop/ros2_env/uav_ws/install/setup.bash"
  elif [[ -f "$REPO_ROOT/install/setup.bash" ]]; then
    # shellcheck source=/dev/null
    source "$REPO_ROOT/install/setup.bash"
  else
    echo "UYARI: setup.bash bulunamadı. ROUTE_HEALTH_WS_SETUP ile tam yol verin." >&2
  fi
}

source_ros_env

# --- Sadece GIL karşılaştırması: 3 topic paralel, ~GIL_CHECK_SEC saniye toplam süre ---
if [[ "$MODE" == "gil-check" ]]; then
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "================================================================================"
  echo "GIL / EĞİTİM YÜKÜ — 3 TOPİK PARALEL HZ"
  echo "Tarih: $(date -Iseconds) | Host: $(hostname) | ROS_DISTRO: ${ROS_DISTRO:-?}"
  echo "Süre: ${GIL_CHECK_SEC}s (topic başına, paralel) | -w ${GIL_CHECK_WINDOW}"
  echo "Log (tee -a): $LOG_FILE"
  echo ""
  echo "Önce: use_sim_time + EKF ayarlarını uygula (Claude Adım 1–2)."
  echo "Karşılaştırma: train KAPALI iken bir kez çalıştır; train AÇIK (~20 sn sonra)"
  echo "aynı dosyaya ikinci kez:  ./scripts/route_pipeline_health_check.sh gil-check BU_DOSYA"
  echo "Alarm (kabaca): /odometry/filtered <5 Hz | /threat/state_vec <2 Hz | patch <5 Hz → GIL şüphesi"
  echo "================================================================================"
  tmpdir=$(mktemp -d)
  # shellcheck disable=SC2064
  trap 'rm -rf "$tmpdir"' EXIT
  SEC="$GIL_CHECK_SEC"
  WIN="$GIL_CHECK_WINDOW"
  echo ""
  echo "\$ timeout ${SEC}s ros2 topic hz ... (3 paralel, wait)"
  timeout "$SEC" ros2 topic hz /odometry/filtered -w "$WIN" >"$tmpdir/odom.txt" 2>&1 &
  p1=$!
  timeout "$SEC" ros2 topic hz /threat/state_vec -w "$WIN" >"$tmpdir/threat.txt" 2>&1 &
  p2=$!
  timeout "$SEC" ros2 topic hz /route/costmap_patch -w "$WIN" >"$tmpdir/patch.txt" 2>&1 &
  p3=$!
  wait $p1 $p2 $p3
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " /odometry/filtered"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cat "$tmpdir/odom.txt"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " /threat/state_vec"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cat "$tmpdir/threat.txt"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " /route/costmap_patch"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cat "$tmpdir/patch.txt"
  echo ""
  echo "================================================================================"
  echo "Bitti. İpucu: grep 'average rate' $LOG_FILE | tail -20"
  echo "================================================================================"
  exit 0
fi

exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "ROUTE PIPELINE HEALTH CHECK"
echo "Tarih (UTC yerel): $(date -Iseconds)"
echo "Hostname: $(hostname)"
echo "ROS_DISTRO: ${ROS_DISTRO:-<yok>}"
echo "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-<varsayılan>}"
echo "Log dosyası: $LOG_FILE"
echo "Hz örnekleme süresi: ${HZ_SAMPLE_SEC}s (topic başına)"
echo "GIL karşılaştırması için: $0 gil-check [aynı_log.txt]"
echo "================================================================================"

section() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " $*"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

run_labeled() {
  local label="$1"
  shift
  echo ""
  echo ">>> $label"
  echo ">>> \$ $*"
  eval "$@"
  echo "--- (exit: $?)"
}

# ros2 topic hz — timeout ile; pencere istatistiği (desteklenmiyorsa -w kaldırılabilir)
sample_hz() {
  local topic="$1"
  echo ""
  echo ">>> HZ ~${HZ_SAMPLE_SEC}s: ${topic}"
  timeout "$HZ_SAMPLE_SEC" ros2 topic hz "$topic" -w 20 2>&1 || echo "(hz: timeout, topic yok veya hz desteklenmiyor — son exit: $?)"
}

topic_info_verbose() {
  local topic="$1"
  echo ""
  echo ">>> ros2 topic info -v $topic"
  ros2 topic info -v "$topic" 2>&1 || echo "(topic info başarısız)"
}

param_try() {
  local node="$1"
  local param="$2"
  echo ""
  echo ">>> ros2 param get ${node} ${param}"
  if ros2 param get "$node" "$param" 2>&1; then
    :
  else
    echo "(node '${node}' yok veya parametre erişilemedi)"
  fi
}

section "A — Sim + köprü"
sample_hz /clock
sample_hz /odometry/filtered
sample_hz /fmu/out/vehicle_odometry
topic_info_verbose /odometry/filtered

section "B — LiDAR (ilk bulunan topic ölçülür)"
LIDAR_CANDIDATES=(
  "/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan"
  "/uav/lidar_2d_v2/scan"
  "/scan"
)
FOUND_LIDAR=""
for t in "${LIDAR_CANDIDATES[@]}"; do
  if ros2 topic list 2>/dev/null | grep -Fxq "$t"; then
    FOUND_LIDAR="$t"
    break
  fi
done
if [[ -n "$FOUND_LIDAR" ]]; then
  echo "Bulunan LiDAR topic: $FOUND_LIDAR"
  sample_hz "$FOUND_LIDAR"
else
  echo "LiDAR: aday topic'lerden hiçbiri listede yok. ros2 topic list | grep -i lidar ile kontrol edin."
  run_labeled "topic list (lidarl)" "ros2 topic list 2>&1 | grep -i lidar || true"
fi
run_labeled "node list (bridge)" "ros2 node list 2>&1 | grep -i bridge || true"

section "C — Nav2 / costmap"
sample_hz /local_costmap/costmap
sample_hz /global_costmap/costmap
topic_info_verbose /local_costmap/costmap
echo ""
echo ">>> Not: TRANSIENT_LOCAL costmap için 'ros2 topic echo' varsayılan QoS ile boş görünebilir."
echo ">>> Örnek (Humble+): ros2 topic echo /local_costmap/costmap --qos-durability transient_local --once"

section "D — Rota özel"
sample_hz /route/costmap_patch
topic_info_verbose /route/costmap_patch
sample_hz /plan
run_labeled "plan --once (içerik)" "timeout 5 ros2 topic echo /plan --once 2>&1 || true"
sample_hz /fmu/in/trajectory_setpoint
run_labeled "goal_pose hz" "timeout $HZ_SAMPLE_SEC ros2 topic hz /goal_pose -w 10 2>&1 || true"

section "E — Threat zinciri"
run_labeled "YOLO detections hz" "timeout $HZ_SAMPLE_SEC ros2 topic hz /yolo/detections -w 15 2>&1 || true"
run_labeled "projected detections hz" "timeout $HZ_SAMPLE_SEC ros2 topic hz /yolo/projected_detections -w 15 2>&1 || true"
topic_info_verbose /threat/state_vec
sample_hz /threat/state_vec
topic_info_verbose /threat/target_scores
sample_hz /threat/target_scores

echo ""
echo ">>> use_sim_time (repoda node adları: threat_encoder_node, threat_target_node)"
param_try /threat_encoder_node use_sim_time
param_try /threat_target_node use_sim_time
# Eski/alternatif isimler (bazı launch'larda remap)
param_try /threat_encoder_v2 use_sim_time
param_try /target_node use_sim_time

section "F — Env / eğitim ile ilgili topic'ler"
sample_hz /route/waypoint_desired
run_labeled "route_curriculum env node var mı" "ros2 node list 2>&1 | grep -E 'route_curriculum|curriculum_env' || true"

section "G — Özet: zorunlu topic var mı?"
check_exists() {
  local t="$1"
  if ros2 topic list 2>/dev/null | grep -Fxq "$t"; then
    echo "  [VAR]  $t"
  else
    echo "  [YOK]  $t"
  fi
}
echo ""
for t in clock odometry/filtered threat/state_vec threat/target_scores route/costmap_patch \
       local_costmap/costmap global_costmap/costmap plan goal_pose \
       fmu/in/trajectory_setpoint; do
  check_exists "/$t"
done

section "BİTTİ"
echo "Rapor kaydedildi: $LOG_FILE"
echo "================================================================================"
