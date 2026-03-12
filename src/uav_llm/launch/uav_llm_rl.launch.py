from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        Node(
            package='uav_threat_agent',
            executable='target_node',
            name='target_node',
            output='screen',
            emulate_tty=True, 
            parameters=[
                {'use_sim_time': True} 
            ]
        ),

        Node(
            package='uav_llm',
            executable='state_summarizer_node',
            name='state_summarizer_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'use_sim_time': True}
            ]
        ),

    
        Node(
            package='uav_llm',
            executable='llm_node',
            name='llm_strategic_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'use_sim_time': True}
            ]
        ),
       
    ])