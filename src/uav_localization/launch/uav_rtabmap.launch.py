from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Verify this path matches your actual file location
    parameters_file_path = '/home/ubuntu/Desktop/ros2_env/uav_ws/src/uav_localization/config/rtabmap_uav.yaml'
    lidar_topic = '/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan/points'
    # 1. Locate the nav2_bringup package
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    # 2. Define the Path to your params file
    # (Using the absolute path you provided to be safe)
    nav2_params_path = '/home/ubuntu/Desktop/ros2_env/uav_ws/src/uav_localization/config/nav2_params.yaml'

    # 3. Include the Nav2 Launch File
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'autostart': 'true',
            'params_file': nav2_params_path
        }.items()
    )


    return LaunchDescription([
        nav2_launch,
        # 1. Main SLAM Node (Package is now 'rtabmap_slam')
        Node(
            package='rtabmap_slam',  # <--- CHANGED FROM rtabmap_ros
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[
                parameters_file_path,
                {'use_sim_time': True}
            ],
            remappings=[
                ('odom', '/odometry/filtered'),
                # ('odom', '/rtabmap/odom'),
                ('scan', '/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan'),
                # ('imu', '/world/default/model/x500_mono_cam_0/link/base_link/sensor/imu_sensor/imu'),
                # ('rgb/image', '/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image'),
                # ('rgb/camera_info', '/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/camera_info')
            ],
            arguments=['--delete_db_on_start']
        ),

        # 2. Visualization Node (Package is now 'rtabmap_viz')
        Node(
            package='rtabmap_viz',   # <--- CHANGED FROM rtabmap_ros
            executable='rtabmap_viz',
            output='screen',
            parameters=[parameters_file_path],
            remappings=[
                ('odom', '/odometry/filtered'),
                ('scan', '/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan'),
                # ('imu', '/world/default/model/x500_mono_cam_0/link/base_link/sensor/imu_sensor/imu'),
                # ('rgb/image', '/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image'),
                # ('rgb/camera_info', '/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/camera_info')
            ]
        ),

        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[{
                'resolution': 0.05,
                'frame_id': 'map',
                'base_frame_id': 'base_link',
                'sensor_model/max_range': 5.0,
                'latch': False,   # Update topics immediately
                'filter_ground': False # Keep the floor (useful for drones)
            }],
            remappings=[
                # Use the Map Cloud from RTAB-Map (PointCloud2)
                # This fixes the LaserScan vs PointCloud2 mismatch
                ('cloud_in', lidar_topic)
            ]
        ),
    ])