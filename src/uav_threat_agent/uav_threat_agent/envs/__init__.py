from gymnasium.envs.registration import register
from uav_threat_agent.envs.threat_env import ThreatAgentEnv

# Ortamı Gym'e kaydediyoruz
register(
    id='ThreatAgent-v0',
    entry_point='uav_threat_agent.envs.threat_env:ThreatAgentEnv',
    max_episode_steps=100, # İsteğe bağlı: Bir bölüm en fazla kaç adım sürsün
)