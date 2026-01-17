from setuptools import find_packages, setup

package_name = 'uav_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/uav_localization']),
    ('share/uav_localization', ['package.xml']),
    ('share/uav_localization/launch',
        ['launch/uav_slam.launch.py']),
    ('share/uav_localization/launch',
        ['launch/uav_rtabmap.launch.py']),
    ('share/uav_localization/config',
        ['config/slam_mapping.yaml']),
    ('share/uav_localization/config',
        ['config/rtabmap_uav.yaml']),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='samwnchstrgl@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "scan_filtered = uav_localization.scan_filtered:main",
        ],
    },
)
