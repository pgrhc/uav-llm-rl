from setuptools import find_packages
from setuptools import setup

setup(
    name='fusion_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('fusion_msgs', 'fusion_msgs.*')),
)
