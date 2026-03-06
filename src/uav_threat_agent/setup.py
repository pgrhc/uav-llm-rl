from setuptools import find_packages, setup

package_name = 'uav_threat_agent'

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
            'train = uav_threat_agent.nodes.train_debug_node:main',
            'inference = uav_threat_agent.nodes.threat_agent_node:main',
            'pretrained_train = uav_threat_agent.nodes.pretrained_model_node:main',
            "curriculum_train = uav_threat_agent.nodes.curriculum_training:main",
            "train_v2 = uav_threat_agent.nodes.train_v2:main",
            "pretrain_v2 = uav_threat_agent.nodes.pretrain_v2:main",
            "target_node = uav_threat_agent.nodes.target_node:main",
        ],
    },
)
