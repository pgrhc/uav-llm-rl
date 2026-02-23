#!/usr/bin/env python3
"""
Custom feature extractor for Route Planning Agent.

Architecture:
    costmap_patch (1,64,64) → CNN  → map_embedding   (128)
    threat_vector (88,)     → MLP  → threat_embedding (64)
    threat_scores (5,)      → MLP  → score_embedding  (16)  ← tehdit ajanı çıktısı
    goal_state    (7,)      → MLP  → goal_embedding   (32)
                                         │
                                    concat (240)
                                         │
                              shared layers (policy_kwargs.net_arch)
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RouteCombinedExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible combined extractor.
    Produces a flat feature vector from Dict observation with
    costmap_patch, threat_vector and goal_state keys.
    """

    def __init__(self, observation_space: spaces.Dict):
        # Total output features = 128 + 64 + 16 + 32 = 240
        super().__init__(observation_space, features_dim=240)

        # --- CNN for costmap_patch (1, 64, 64) ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),   # → (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # → (64, 8, 8)
            nn.ReLU(),
            nn.Flatten(),                                           # → 64*8*8 = 4096
        )
        self.cnn_fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
        )

        # --- MLP for threat_vector (88,) ---
        self.threat_mlp = nn.Sequential(
            nn.Linear(88, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # --- MLP for threat_scores (5,) — tehdit ajanı çıktısı ---
        self.scores_mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
        )

        # --- MLP for goal_state (7,) ---
        self.goal_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        costmap = observations["costmap_patch"]     # (B, 1, 64, 64)
        threat = observations["threat_vector"]       # (B, 88)
        scores = observations["threat_scores"]       # (B, 5)
        goal = observations["goal_state"]            # (B, 7)

        map_emb = self.cnn_fc(self.cnn(costmap))     # (B, 128)
        threat_emb = self.threat_mlp(threat)         # (B, 64)
        score_emb = self.scores_mlp(scores)          # (B, 16)
        goal_emb = self.goal_mlp(goal)              # (B, 32)

        return torch.cat([map_emb, threat_emb, score_emb, goal_emb], dim=1)  # (B, 240)
