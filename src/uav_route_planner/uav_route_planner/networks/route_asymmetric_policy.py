#!/usr/bin/env python3
"""
Route Asymmetric SAC Policy

Actor:  Deployable obs only (costmap, threat, scores, goal, path) -> 272
Critic: Deployable + privileged -> 272 + PRIV_DIM

Inference: Actor uses only deployable; privileged can be zeros.
"""

from __future__ import annotations

from typing import List, Optional

import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import SACPolicy, Actor, ContinuousCritic

from uav_route_planner.networks.route_extractor import RouteCombinedExtractor


class RouteActorExtractor(RouteCombinedExtractor):
    """Actor: deployable obs only. Ignores 'privileged' key."""

    def forward(self, observations: dict) -> th.Tensor:
        return super().forward(observations)


class RouteCriticExtractor(RouteCombinedExtractor):
    """Critic: deployable (272) + privileged concat. MRO: super().forward -> RouteCombinedExtractor.forward."""

    def __init__(self, observation_space: spaces.Dict):
        priv_dim = int(observation_space["privileged"].shape[0])
        super().__init__(
            observation_space,
            features_dim=RouteCombinedExtractor.DEPLOY_DIM + priv_dim,
        )
        self._priv_dim = priv_dim

    def forward(self, observations: dict) -> th.Tensor:
        deploy_emb = super().forward(observations)
        if deploy_emb.shape[-1] != RouteCombinedExtractor.DEPLOY_DIM:
            raise RuntimeError(
                f"RouteCombinedExtractor expected {RouteCombinedExtractor.DEPLOY_DIM}, "
                f"got {deploy_emb.shape[-1]}"
            )
        priv = observations["privileged"].float()
        return th.cat([deploy_emb, priv], dim=1)


class RouteAsymmetricSACPolicy(SACPolicy):
    """SAC with asymmetric obs: Actor sees deployable only, Critic sees deployable+privileged."""

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
        kwargs["features_extractor_class"] = RouteActorExtractor
        kwargs["features_extractor_kwargs"] = {}
        kwargs["share_features_extractor"] = False
        kwargs.setdefault("net_arch", [])

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_actor(self, features_extractor=None) -> Actor:
        actor_fe = RouteActorExtractor(self.observation_space)
        kwargs = self._update_features_extractor(self.actor_kwargs, actor_fe)
        kwargs["net_arch"] = self._pi_arch
        return Actor(**kwargs).to(self.device)

    def make_critic(self, features_extractor=None) -> ContinuousCritic:
        critic_fe = RouteCriticExtractor(self.observation_space)
        kwargs = self._update_features_extractor(self.critic_kwargs, critic_fe)
        kwargs["net_arch"] = self._vf_arch
        return ContinuousCritic(**kwargs).to(self.device)

    @th.no_grad()
    def predict_actor_only(
        self,
        obs_dict: dict,
        deterministic: bool = True,
    ) -> th.Tensor:
        """Inference: Actor only. Pass obs with privileged=zeros if not available."""
        priv = obs_dict.get("privileged")
        if priv is None:
            costmap = obs_dict["costmap_patch"]
            batch = costmap.shape[0] if hasattr(costmap, "shape") else 1
            device = getattr(costmap, "device", self.device)
            dtype = getattr(costmap, "dtype", th.float32)
            priv = th.zeros(
                batch,
                self.observation_space["privileged"].shape[0],
                device=device,
                dtype=dtype,
            )
            obs_dict = {**obs_dict, "privileged": priv}
        obs_tensor, _ = self.obs_to_tensor(obs_dict)
        actions, _ = self._predict(obs_tensor, deterministic=deterministic)
        return actions
