#!/usr/bin/env python3
"""
Custom feature extractor for Route Planning Agent  (Faz 1).

Uses only deployable keys: costmap_patch, threat_vector, threat_scores, goal_state, a_star_path.
Ignores 'privileged' if present in observations (for Actor; Critic adds it separately).

Architecture:
    lidar_vector  (36,)      → MLP  → map_embedding    (128)
    threat_vector (74,)      → MLP  → threat_embedding (64)
    threat_scores (5,)       → MLP  → score_embedding  (16)
    goal_state    (7,)       → MLP  → goal_embedding   (32)
    a_star_path   (10,)      → MLP  → path_embedding   (32)
                                          │
                                     concat (272)
                                          │
                               shared layers (net_arch)
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Optional

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RouteCombinedExtractor(BaseFeaturesExtractor):

    DEPLOY_DIM = 272

    def __init__(self, observation_space: spaces.Dict, features_dim: Optional[int] = None):
        if features_dim is not None and features_dim <= 0:
            raise ValueError("features_dim must be > 0")
        fd = features_dim if features_dim is not None else self.DEPLOY_DIM
        super().__init__(observation_space, features_dim=fd)

        # MLP for lidar_vector (36,)
        self.lidar_mlp = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
        )

        # MLP for threat_vector (74,)
        self.threat_mlp = nn.Sequential(
            nn.Linear(74, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # MLP for threat_scores (5,)
        self.scores_mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
        )

        # MLP for goal_state (7,)
        self.goal_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # MLP for a_star_path (10,)
        self.path_mlp = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        """Extract deployable features only. Ignores 'privileged' if present."""
        lidar = observations["lidar_vector"]
        threat = observations["threat_vector"]
        scores = observations["threat_scores"]
        goal = observations["goal_state"]
        path = observations["a_star_path"]

        map_emb = self.lidar_mlp(lidar)                # (B, 128)
        threat_emb = self.threat_mlp(threat)           # (B, 64)
        score_emb = self.scores_mlp(scores)            # (B, 16)
        goal_emb = self.goal_mlp(goal)                 # (B, 32)
        path_emb = self.path_mlp(path)                 # (B, 32)

        return torch.cat(
            [map_emb, threat_emb, score_emb, goal_emb, path_emb], dim=1
        )  # (B, 272)
