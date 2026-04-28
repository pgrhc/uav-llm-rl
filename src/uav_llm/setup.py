from setuptools import find_packages, setup

package_name = 'uav_llm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/uav_llm/launch',
        ['launch/uav_llm_rl.launch.py']),
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
            "llm_node = uav_llm.llm_strategic_node:main",
            "state_summarizer_node = uav_llm.state_summarizer_node:main",
            "llm_task_node = uav_llm.llm_task:main",
            "llm_executor_node = uav_llm.llm_executor_node:main",
            "llm_command_input_node = uav_llm.llm_command_input_node:main",
            "llm_raw_task_node = uav_llm.llm_raw_task:main",
            "plan_review_node = uav_llm.llm_plan_review:main",
            "plan_execution_node = uav_llm.llm_plan_execution:main",
            "user_command_node = uav_llm.user_command_node:main",
            "mission_interpreter_node = uav_llm.mission_interpreter_node:main",
            "task_planner_node = uav_llm.task_planner_node:main",
            "semantic_object_builder_node = uav_llm.semantic_object_builder_node:main",
            "semantic_memory_node = uav_llm.semantic_memory_node:main",
            "llm_scene_summary_node = uav_llm.llm_scene_summary_node:main",
        ],
    },
)
