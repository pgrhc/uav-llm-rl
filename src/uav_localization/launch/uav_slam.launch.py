import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('uav_localization')
    slam_params_file = os.path.join(pkg_share, 'config', 'slam_mapping.yaml')

    # Senin uzun sensör topic ismin (okunabilirlik için değişkene atadım)
    lidar_topic = '/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan'

    return LaunchDescription([
        # 1. SLAM Toolbox Node (Mevcut kodun)
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[slam_params_file],
        ),
        

        # 2. Octomap Server Node (Eklenen kısım)
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[{
                'resolution': 0.05,
                'frame_id': 'map',
                'base_frame_id': 'base_link',
                # Gerekirse sensör modeline göre max_range vb. ekleyebilirsin
                # 'sensor_model/max_range': 5.0 
            }],
            remappings=[
                ('cloud_in', lidar_topic)
            ]
        )
    ])