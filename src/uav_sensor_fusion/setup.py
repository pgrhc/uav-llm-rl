from setuptools import find_packages, setup

package_name = 'uav_sensor_fusion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/uav_sensor_fusion/launch', ['launch/gz_bridge.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "px4_odom_bridge = uav_sensor_fusion.px4_odom_bridge:main",
            "lidar = uav_sensor_fusion.lidar:main",
            "radar = uav_sensor_fusion.radar:main",
            "radar_filter = uav_sensor_fusion.radar_filter:main",
            "cloud_saver = uav_sensor_fusion.cloud_saver:main",
            "sensor_fusion_sync = uav_sensor_fusion.sensor_fusion_sync:main",
            "bev_node = uav_sensor_fusion.bev_node:main",
            "yolo_node = uav_sensor_fusion.yolo_node:main",
            "bev_projection_node = uav_sensor_fusion.bev_projection_node:main",
        ],
    },
)
