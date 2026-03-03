from gymnasium.envs.registration import register
from uav_threat_agent.envs.curriculum_env import ThreatAgentEnv

# Ortamı Gym'e kaydediyoruz
register(
    id='ThreatAgent-v11',
    entry_point='uav_threat_agent.envs.curriculum_env:ThreatAgentEnv',
    max_episode_steps=4096, # İsteğe bağlı: Bir bölüm en fazla kaç adım sürsün
)