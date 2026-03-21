#!/bin/bash
# Route training için gerekli topic'lerin yayınlanıp yayınlanmadığını kontrol eder
# Kullanım: ./scripts/check_route_topics.sh

echo "=== Route Training Topic Kontrolü ==="
echo ""

check_topic() {
    if ros2 topic list | grep -q "^/$1$"; then
        hz=$(timeout 2 ros2 topic hz "$1" 2>/dev/null | grep "average rate" || echo "? Hz")
        echo "✓ $1 — $hz"
    else
        echo "✗ $1 — YOK"
    fi
}

echo "Zorunlu topic'ler:"
check_topic "route/costmap_patch"
check_topic "threat/state_vec"
check_topic "threat/target_scores"
check_topic "odometry/filtered"
check_topic "goal_pose"
check_topic "plan"
check_topic "route/waypoint_desired"
check_topic "local_costmap/costmap"
check_topic "global_costmap/costmap"

echo ""
echo "PX4 setpoint:"
check_topic "fmu/in/trajectory_setpoint"

echo ""
echo "Eksik topic varsa ilgili node'u başlatın:"
echo "  costmap_patch_node, threat_encoder_v2, target_node, route_goal_navigator, follow_path"
