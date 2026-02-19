from setuptools import find_packages, setup

package_name = 'drone_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            "offboard_control = drone_control.offboard_control:main",
            "q_learn = drone_control.q_learn:main",
            "train = drone_control.train:main",
            "maze_navigator = drone_control.maze_navigator:main",
            "follow_path = drone_control.follow_path:main",
            "go_to_maze_exit = drone_control.go_to_maze_exit:main",
            "auto_maze_navigator = drone_control.auto_maze_navigator:main",
            "drone_navigator = drone_control.drone_navigator:main",
        ],
    },
)
