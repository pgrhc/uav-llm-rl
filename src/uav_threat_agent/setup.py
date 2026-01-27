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
            # Canlı ajanı çalıştırmak için (threat_agent_node.py içindeki main fonksiyonu)
            'inference = uav_threat_agent.nodes.threat_agent_node:main',
        ],
    },
)
