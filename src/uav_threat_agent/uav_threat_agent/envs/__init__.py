from gymnasium.envs.registration import register
from uav_threat_agent.envs.three_layer_threat_env import ThreatAgentEnv

# Ortamı Gym'e kaydediyoruz
register(
    id='ThreatAgent-v6',
    entry_point='uav_threat_agent.envs.three_layer_threat_env:ThreatAgentEnv',
    max_episode_steps=200, # İsteğe bağlı: Bir bölüm en fazla kaç adım sürsün
)