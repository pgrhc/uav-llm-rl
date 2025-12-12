from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.actions import SetParameter

ld = LaunchDescription()
ld.add_action(SetParameter(name="use_sim_time", value=True))


def generate_launch_description():

    # 1) Gazebo IMU -> ROS (sensor_msgs/Imu)
    imu_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_imu_bridge",
        output="screen",
        arguments=[
            "/world/default/model/x500_mono_cam_0/link/base_link/sensor/imu_sensor/imu"
            "@sensor_msgs/msg/Imu"
            "@gz.msgs.IMU"
        ],
    )
    # 2) Kamera: Gazebo Image -> ROS Image
    camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_camera_bridge",
        output="screen",
        arguments=[
            "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"
            "@sensor_msgs/msg/Image"
            "@gz.msgs.Image"
        ],
    )
    camera_info_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_camera_info_bridge",
        output="screen",
        arguments=[
            "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/camera_info"
            "@sensor_msgs/msg/CameraInfo"
            "@gz.msgs.CameraInfo"
        ],
    )
        # LiDAR: LaserScan
    lidar_scan_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_lidar_scan_bridge",
        output="screen",
        arguments=[
            "/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan"
            "@sensor_msgs/msg/LaserScan"
            "@gz.msgs.LaserScan"
        ],
    )

    # LiDAR: PointCloud2 (x,y,z için)
    lidar_points_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_lidar_points_bridge",
        output="screen",
        arguments=[
            "/world/default/model/x500_mono_cam_0/link/link/sensor/lidar_2d_v2/scan/points"
            "@sensor_msgs/msg/PointCloud2"
            "@gz.msgs.PointCloudPacked"
        ],
    )

    radar_points_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_radar_points_bridge",
        output="screen",
        arguments=[
            "/radar/points"
            "@sensor_msgs/msg/PointCloud2"
            "@gz.msgs.PointCloudPacked"
        ],
    )

    # 2) FMU vehicle_odometry -> /odom/px4 (nav_msgs/Odometry)
    px4_odom_bridge = Node(
        package="uav_sensor_fusion",
        executable="px4_odom_bridge",
        name="px4_odom_bridge",
        output="screen",
    )

    # 3) EKF (robot_localization)
    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            
            "/home/ubuntu/Desktop/ros2_env/uav_ws/config/ekf_uav.yaml"
        ],
    )
    radar_filter = Node(
    package="uav_sensor_fusion",
    executable="radar_filter",
    name="radar_filter",
    output="screen",
    parameters=[{'use_sim_time': True}]
    )
    # RADAR → BASE LINK TF
    radar_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="radar_static_tf",
        arguments=["0.15", "0", "0.12", "0", "0", "0", "base_link", "radar_link"]
    )
    lidar_tf = Node(
    package="tf2_ros",
    executable="static_transform_publisher",
    arguments=["0", "0", "0.084", "0", "0", "0", "base_link", "link"]
    )
    camera_tf = Node(
    package="tf2_ros",
    executable="static_transform_publisher",
    name="camera_static_tf",
    arguments=["0.12", "0.03", "0.242", "0", "0", "0", "base_link", "camera_link"]
)

    clock_bridge = Node(
    package="ros_gz_bridge",
    executable="parameter_bridge",
    name="clock_bridge",
    output="screen",
    arguments=[
        "/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock"
    ]
)
    sync_node = Node(
        package="uav_sensor_fusion",
        executable="sensor_fusion_sync",
        name="sensor_fusion_sync",
        output="screen",
        parameters=[{'use_sim_time': True}]
    )


    yolo_node = Node(
        package="uav_sensor_fusion",
        executable="yolo_node",
        name="yolo_node",
        output="screen",
        parameters=[{'use_sim_time': True}]
    )
    bev_proj_node = Node(
        package="uav_sensor_fusion",
        executable="bev_projection_node",
        name="bev_projection_node",
        output="screen",
        parameters=[{'use_sim_time': True}]
    )
    
    return LaunchDescription([
        imu_bridge,
        camera_bridge,
        camera_info_bridge,
        lidar_scan_bridge,
        lidar_points_bridge,
        radar_points_bridge,
        px4_odom_bridge,
        ekf_node,
        radar_filter, 
        lidar_tf,
        camera_tf,
        radar_tf,
        clock_bridge,
        sync_node,
        yolo_node,
        bev_proj_node,
        
    ])