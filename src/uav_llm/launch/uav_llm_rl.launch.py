from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # ---------------------------------------------------------
        # 1. THREAT AGENT INFERENCE (Tehdit Algılama - RL Modeli)
        # ---------------------------------------------------------
        # Paket: uav_threat_agent
        # Executable: inference (setup.py'da tanımladığın isim)
        Node(
            package='uav_threat_agent',
            executable='inference',
            name='threat_agent_node',
            output='screen',
            emulate_tty=True, # Renkli loglar için
            parameters=[
                {'use_sim_time': True} # Simülasyon zamanı ile senkronize ol
            ]
        ),

        # ---------------------------------------------------------
        # 2. STATE SUMMARIZER (Veri Füzyonu ve JSON Dönüşümü)
        # ---------------------------------------------------------
        # Paket: uav_llm
        # Executable: state_summarizer (setup.py'da tanımladığın isim)
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

        # ---------------------------------------------------------
        # 3. LLM STRATEGIC NODE (Karar Mekanizması - CoT)
        # ---------------------------------------------------------
        # Paket: uav_llm
        # Executable: llm_node (setup.py'da tanımladığın isim)
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