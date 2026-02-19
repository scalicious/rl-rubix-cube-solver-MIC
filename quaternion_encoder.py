"""
quaternion_encoder.py - Encodes cube state into a 270-dim vector using
quaternion representations of cubie rotations.

Raw encoding (288 dims):
  - 8 corners x 4 (quaternion) = 32
  - 8 corners x 8 (one-hot position) = 64
  - 12 edges x 4 (quaternion) = 48
  - 12 edges x 12 (one-hot position) = 144
Then projected to 270 via a learned linear layer.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional


# Quaternion helpers

def quat_identity():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def quat_multiply(q1, q2):
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def quat_normalize(q):
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return quat_identity()
    return q / norm


# Rotation quaternions for cubie orientations

# 3 twist states for corners (identity, 120 CW, 240 CW)
CORNER_ORIENT_QUATS = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5, -0.5],
], dtype=np.float32)

# 2 flip states for edges
EDGE_ORIENT_QUATS = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
], dtype=np.float32)

# Where each cubie lives in 3D when solved (normalized)
CORNER_HOME_POSITIONS = np.array([
    [-1, +1, -1],  # ULB
    [+1, +1, -1],  # UBR
    [+1, +1, +1],  # URF
    [-1, +1, +1],  # UFL
    [-1, -1, +1],  # DLF
    [+1, -1, +1],  # DFR
    [+1, -1, -1],  # DRB
    [-1, -1, -1],  # DBL
], dtype=np.float32) / np.sqrt(3.0)

EDGE_HOME_POSITIONS = np.array([
    [0, +1, -1],   # UB
    [+1, +1, 0],   # UR
    [0, +1, +1],   # UF
    [-1, +1, 0],   # UL
    [-1, 0, +1],   # FL
    [+1, 0, +1],   # FR
    [-1, 0, -1],   # BL
    [+1, 0, -1],   # BR
    [0, -1, +1],   # DF
    [+1, -1, 0],   # DR
    [0, -1, -1],   # DB
    [-1, -1, 0],   # DL
], dtype=np.float32) / np.sqrt(2.0)


class QuaternionStateEncoder:
    """
    Converts cube state dict into a 288-dim numpy vector:
    corner quaternions + corner positions + edge quaternions + edge positions.
    """

    RAW_DIM = 288

    def encode(self, state):
        corner_perm = state['corner_perm']
        corner_orient = state['corner_orient']
        edge_perm = state['edge_perm']
        edge_orient = state['edge_orient']

        parts = []

        # corner quaternions (8 x 4 = 32)
        for i in range(8):
            orient = int(corner_orient[i])
            parts.append(CORNER_ORIENT_QUATS[orient])

        # corner position one-hots (8 x 8 = 64)
        for i in range(8):
            one_hot = np.zeros(8, dtype=np.float32)
            one_hot[int(corner_perm[i])] = 1.0
            parts.append(one_hot)

        # edge quaternions (12 x 4 = 48)
        for i in range(12):
            orient = int(edge_orient[i])
            parts.append(EDGE_ORIENT_QUATS[orient])

        # edge position one-hots (12 x 12 = 144)
        for i in range(12):
            one_hot = np.zeros(12, dtype=np.float32)
            one_hot[int(edge_perm[i])] = 1.0
            parts.append(one_hot)

        return np.concatenate(parts)

    def encode_batch(self, states):
        return np.stack([self.encode(s) for s in states])

    def compute_geometric_features(self, state):
        """
        Euclidean distance between each cubie's current slot and home slot.
        Returns (20,) — 8 corners + 12 edges.
        """
        corner_perm = state['corner_perm']
        edge_perm = state['edge_perm']

        distances = np.zeros(20, dtype=np.float32)

        for slot in range(8):
            cubie = int(corner_perm[slot])
            distances[slot] = np.linalg.norm(
                CORNER_HOME_POSITIONS[slot] - CORNER_HOME_POSITIONS[cubie]
            )

        for slot in range(12):
            cubie = int(edge_perm[slot])
            distances[8 + slot] = np.linalg.norm(
                EDGE_HOME_POSITIONS[slot] - EDGE_HOME_POSITIONS[cubie]
            )

        return distances


class QuaternionEncoder(nn.Module):
    """
    Wraps raw quaternion encoding + learned projection to 270 dims.
    Also integrates geometric distance features via a side branch.
    """

    OUTPUT_DIM = 270

    def __init__(self):
        super().__init__()
        self.raw_encoder = QuaternionStateEncoder()

        # 288 -> 270 projection
        self.projection = nn.Sequential(
            nn.Linear(QuaternionStateEncoder.RAW_DIM, 270),
            nn.LayerNorm(270),
            nn.GELU(),
        )

        # geometric features (20 -> 270)
        self.geo_projection = nn.Sequential(
            nn.Linear(20, 64),
            nn.GELU(),
            nn.Linear(64, 270),
        )

        # combine main + geo
        self.combine = nn.Sequential(
            nn.Linear(270 * 2, 270),
            nn.LayerNorm(270),
            nn.GELU(),
        )

    def encode_state(self, state):
        return self.raw_encoder.encode(state)

    def encode_state_batch(self, states):
        return self.raw_encoder.encode_batch(states)

    def forward(self, raw_encoding, geo_features=None):
        """
        Project raw encoding (288) to 270-dim output.
        If geo_features (20) provided, fuses them in.
        """
        main = self.projection(raw_encoding)

        if geo_features is not None:
            geo = self.geo_projection(geo_features)
            combined = torch.cat([main, geo], dim=-1)
            return self.combine(combined)

        return main

    def encode_and_project(self, states, device):
        """Full pipeline: state dicts -> 270-dim tensor."""
        raw = self.raw_encoder.encode_batch(states)
        raw_tensor = torch.FloatTensor(raw).to(device)

        geo_list = [self.raw_encoder.compute_geometric_features(s) for s in states]
        geo_tensor = torch.FloatTensor(np.stack(geo_list)).to(device)

        return self.forward(raw_tensor, geo_tensor)
