"""
Rota yardımcı yığını: costmap patch + güvenlik filtresi + PX4 follow_path.

Zincir (varsayılan):
  … → /route/waypoint_desired → route_safety_filter → /route/waypoint_safe
  → follow_path (route_waypoint_topic) → /fmu/in/trajectory_setpoint

Filtreyi atlamak için:
  ros2 launch uav_route_planner route_stack.launch.py follow_waypoint_topic:=/route/waypoint_desired

PX4 / offboard çalıştırmayacaksan:
  ros2 launch uav_route_planner route_stack.launch.py launch_follow_path:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "follow_waypoint_topic",
                default_value="/route/waypoint_safe",
                description="follow_path aboneliği: güvenli çıktı veya doğrudan desired",
            ),
            DeclareLaunchArgument(
                "launch_follow_path",
                default_value="true",
                description="false: sadece costmap_patch + route_safety_filter",
            ),
            Node(
                package="uav_route_planner",
                executable="costmap_patch_node",
                name="costmap_patch_node",
            ),
            Node(
                package="uav_route_planner",
                executable="route_safety_filter_node",
                name="route_safety_filter_node",
            ),
            Node(
                package="drone_control",
                executable="follow_path",
                name="follow_path",
                condition=IfCondition(LaunchConfiguration("launch_follow_path")),
                parameters=[
                    {
                        "route_waypoint_topic": ParameterValue(
                            LaunchConfiguration("follow_waypoint_topic"),
                            value_type=str,
                        ),
                    }
                ],
            ),
        ]
    )
