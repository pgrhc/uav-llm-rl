#!/usr/bin/env python3
import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RouteCombinedExtractor(BaseFeaturesExtractor):
    DEPLOY_DIM = 144 

    def __init__(self, observation_space: spaces.Dict, features_dim: Optional[int] = None):
        if features_dim is not None and features_dim <= 0:
            raise ValueError("features_dim must be > 0")

        fd = features_dim if features_dim is not None else self.DEPLOY_DIM
        super().__init__(observation_space, features_dim=fd)

        self.threat_mlp = nn.Sequential(
            nn.Linear(74, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.scores_mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
        )

        # Extractor içinde goal_mlp'yi şu şekilde güçlendirmeyi dene:
        self.goal_mlp = nn.Sequential(
            nn.Linear(7, 128), # Daha geniş giriş
            nn.ReLU(),
            nn.LayerNorm(128), # Veriyi normalize et ki baskın olsun
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        threat = observations["threat_vector"]
        scores = observations["threat_scores"]
        goal = observations["goal_state"]

        threat_emb = self.threat_mlp(threat)
        score_emb = self.scores_mlp(scores)
        goal_emb = self.goal_mlp(goal)
        goal_emb = goal_emb * 1.3
        return torch.cat(
            [threat_emb, score_emb, goal_emb], dim=1
        )