from gymnasium.envs.registration import register
from uav_threat_agent.envs.env_v2 import ThreatAgentEnv

# Ortamı Gym'e kaydediyoruz
register(
    id='ThreatAgent-v13',
    entry_point='uav_threat_agent.envs.env_v3:ThreatAgentEnv',
    max_episode_steps=2048, # İsteğe bağlı: Bir bölüm en fazla kaç adım sürsün
)