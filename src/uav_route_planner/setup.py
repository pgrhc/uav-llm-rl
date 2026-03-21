from setuptools import find_packages, setup

package_name = 'uav_route_planner'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='samwnchstrgl@gmail.com',
    description='RL-based route planning agent with CBF safety filter for UAV navigation',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            # Faz 0: Altyapı node'ları
            'costmap_patch_node = uav_route_planner.costmap_patch_node:main',
            'heuristic_planner_node = uav_route_planner.heuristic_planner_node:main',
            'route_safety_filter_node = uav_route_planner.route_safety_filter_node:main',
            # Faz 1: RL route agent
            'train_route_node = uav_route_planner.nodes.train_route_node:main',
            'route_agent_node = uav_route_planner.nodes.route_agent_node:main',
            # Curriculum training
            'train_route_curriculum = uav_route_planner.nodes.train_route_curriculum:main',
            'route_curriculum_agent_node = uav_route_planner.nodes.route_curriculum_agent_node:main',
            # Mega-world maze spawner
            'maze_curriculum_world = uav_route_planner.maze_curriculum_world:main',
        ],
    },
)
