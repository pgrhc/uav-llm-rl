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
        ['launch/localization.launch.py']),
    ('share/uav_localization/config',
        ['config/slam_localization.yaml']),
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
        ],
    },
)
