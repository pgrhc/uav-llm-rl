from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Type, Any

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import SACPolicy, Actor, ContinuousCritic



class ActorFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        obs_dim = int(observation_space["obs"].shape[0])
        super().__init__(observation_space, features_dim=obs_dim)

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        return observations["obs"].float()


class CriticFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        obs_dim  = int(observation_space["obs"].shape[0])
        priv_dim = int(observation_space["privileged"].shape[0])
        super().__init__(observation_space, features_dim=obs_dim + priv_dim)

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        return th.cat(
            [observations["obs"].float(),
             observations["privileged"].float()],
            dim=1
        )

class AsymmetricSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        pi_arch: Optional[List[int]] = None,
        vf_arch: Optional[List[int]] = None,
        **kwargs,
    ):
        self._pi_arch = pi_arch or [256, 256, 128]
        self._vf_arch = vf_arch or [256, 256, 128]
        kwargs["features_extractor_class"]  = ActorFeaturesExtractor
        kwargs["features_extractor_kwargs"] = {}
        kwargs["share_features_extractor"]  = False
        kwargs.setdefault("net_arch", [])

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)


    def make_actor(self, features_extractor=None) -> Actor:
        actor_fe  = ActorFeaturesExtractor(self.observation_space)
        kwargs    = self._update_features_extractor(self.actor_kwargs, actor_fe)
        kwargs["net_arch"] = self._pi_arch
        return Actor(**kwargs).to(self.device)

    def make_critic(self, features_extractor=None) -> ContinuousCritic:
        critic_fe = CriticFeaturesExtractor(self.observation_space)
        kwargs    = self._update_features_extractor(self.critic_kwargs, critic_fe)
        kwargs["net_arch"] = self._vf_arch
        return ContinuousCritic(**kwargs).to(self.device)

    @th.no_grad()
    def predict_actor_only(
        self,
        obs_vec: th.Tensor,
        deterministic: bool = True,
    ) -> th.Tensor:
        dummy_priv = th.zeros(
            obs_vec.shape[0],
            self.observation_space["privileged"].shape[0],
            device=obs_vec.device, dtype=obs_vec.dtype
        )
        obs_dict = {"obs": obs_vec, "privileged": dummy_priv}
        return self.actor(obs_dict, deterministic=deterministic)