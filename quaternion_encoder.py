"""
quaternion_encoder.py — Quaternion State Encoding for MARVELS
==============================================================
Converts raw cube state into a geometrically meaningful 270-dimensional
vector using quaternion representations of cubie rotations.

Encoding Pipeline:
  1. Corner quaternions:   8 corners × 4 = 32 dims
  2. Corner positions:     8 corners × 8 (one-hot) = 64 dims
  3. Edge quaternions:     12 edges × 4 = 48 dims
  4. Edge positions:       12 edges × 12 (one-hot) = 144 dims
  5. Geometric features:   distance metrics = variable dims
  ─────────────────────────────────────────────────────
  Raw total → projected to 270 via learned linear layer

Novel Aspect:
  Unlike flat one-hot sticker encodings, quaternions capture the
  *geometric structure* of cubie rotations in SO(3), enabling the
  policy network to reason about spatial relationships directly.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional


# ─────────────── Quaternion Arithmetic Helpers ───────────────

def quat_identity() -> np.ndarray:
    """Return identity quaternion [w, x, y, z] = [1, 0, 0, 0]."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return quat_identity()
    return q / norm


# ─────── Rotation quaternions for each orientation ───────

# Corner orientation quaternions (0, 1, 2 rotations around diagonal axis)
# These represent the three possible twist states of a corner cubie.
CORNER_ORIENT_QUATS = np.array([
    [1.0, 0.0, 0.0, 0.0],     # Orientation 0: identity (correct)
    [0.5, 0.5, 0.5, 0.5],     # Orientation 1: 120° CW twist
    [0.5, -0.5, -0.5, -0.5],  # Orientation 2: 240° CW twist (120° CCW)
], dtype=np.float32)

# Edge orientation quaternions (0 or 1 flip)
EDGE_ORIENT_QUATS = np.array([
    [1.0, 0.0, 0.0, 0.0],     # Orientation 0: correct
    [0.0, 1.0, 0.0, 0.0],     # Orientation 1: flipped (180° around edge axis)
], dtype=np.float32)

# Home positions of cubies in 3D space (normalized to unit cube)
# These define where each cubie "should be" when solved.
CORNER_HOME_POSITIONS = np.array([
    [-1, +1, -1],  # Corner 0: ULB
    [+1, +1, -1],  # Corner 1: UBR
    [+1, +1, +1],  # Corner 2: URF
    [-1, +1, +1],  # Corner 3: UFL
    [-1, -1, +1],  # Corner 4: DLF
    [+1, -1, +1],  # Corner 5: DFR
    [+1, -1, -1],  # Corner 6: DRB
    [-1, -1, -1],  # Corner 7: DBL
], dtype=np.float32) / np.sqrt(3.0)

EDGE_HOME_POSITIONS = np.array([
    [0, +1, -1],   # Edge 0: UB
    [+1, +1, 0],   # Edge 1: UR
    [0, +1, +1],   # Edge 2: UF
    [-1, +1, 0],   # Edge 3: UL
    [-1, 0, +1],   # Edge 4: FL
    [+1, 0, +1],   # Edge 5: FR
    [-1, 0, -1],   # Edge 6: BL
    [+1, 0, -1],   # Edge 7: BR
    [0, -1, +1],   # Edge 8: DF
    [+1, -1, 0],   # Edge 9: DR
    [0, -1, -1],   # Edge 10: DB
    [-1, -1, 0],   # Edge 11: DL
], dtype=np.float32) / np.sqrt(2.0)


# ────────────────── Raw State Encoder ──────────────────

class QuaternionStateEncoder:
    """
    Encodes a Rubik's Cube state dictionary into a fixed-size numpy vector
    using quaternion representations of cubie rotations and positions.

    Output dimensionality: 288 (before projection)
      - 8 × 4 = 32  (corner quaternions)
      - 8 × 8 = 64  (corner position one-hots)
      - 12 × 4 = 48  (edge quaternions)
      - 12 × 12 = 144 (edge position one-hots)
    """

    RAW_DIM = 288  # Before projection

    def encode(self, state: dict) -> np.ndarray:
        """
        Convert cube state dict to 288-dim numpy vector.

        Args:
            state: dict with 'corner_perm', 'corner_orient',
                   'edge_perm', 'edge_orient'

        Returns:
            np.ndarray of shape (288,)
        """
        corner_perm = state['corner_perm']
        corner_orient = state['corner_orient']
        edge_perm = state['edge_perm']
        edge_orient = state['edge_orient']

        parts = []

        # 1. Corner quaternions (8 × 4 = 32)
        for i in range(8):
            orient = int(corner_orient[i])
            q = CORNER_ORIENT_QUATS[orient]
            parts.append(q)

        # 2. Corner position one-hots (8 × 8 = 64)
        for i in range(8):
            one_hot = np.zeros(8, dtype=np.float32)
            one_hot[int(corner_perm[i])] = 1.0
            parts.append(one_hot)

        # 3. Edge quaternions (12 × 4 = 48)
        for i in range(12):
            orient = int(edge_orient[i])
            q = EDGE_ORIENT_QUATS[orient]
            parts.append(q)

        # 4. Edge position one-hots (12 × 12 = 144)
        for i in range(12):
            one_hot = np.zeros(12, dtype=np.float32)
            one_hot[int(edge_perm[i])] = 1.0
            parts.append(one_hot)

        return np.concatenate(parts)

    def encode_batch(self, states: List[dict]) -> np.ndarray:
        """Encode a batch of states. Returns (batch_size, 288)."""
        return np.stack([self.encode(s) for s in states])

    def compute_geometric_features(self, state: dict) -> np.ndarray:
        """
        Compute geometric distance features for cubies.

        For each cubie, compute the Euclidean distance between its
        current position and its home position. This provides a
        continuous measure of "how far" each piece is from solved.

        Returns: np.ndarray of shape (20,) — 8 corners + 12 edges
        """
        corner_perm = state['corner_perm']
        edge_perm = state['edge_perm']

        distances = np.zeros(20, dtype=np.float32)

        # Corner distances
        for slot in range(8):
            cubie = int(corner_perm[slot])
            current_pos = CORNER_HOME_POSITIONS[slot]
            home_pos = CORNER_HOME_POSITIONS[cubie]
            distances[slot] = np.linalg.norm(current_pos - home_pos)

        # Edge distances
        for slot in range(12):
            cubie = int(edge_perm[slot])
            current_pos = EDGE_HOME_POSITIONS[slot]
            home_pos = EDGE_HOME_POSITIONS[cubie]
            distances[8 + slot] = np.linalg.norm(current_pos - home_pos)

        return distances


# ────────────────── Learned Projection (PyTorch) ──────────────────

class QuaternionEncoder(nn.Module):
    """
    PyTorch module that wraps the raw quaternion encoding and projects
    it to the target dimensionality (270) via a learned linear layer
    with layer normalization.

    This is the main encoder used by all agents in the MARVELS system.

    Architecture:
      Raw encoding (288) → Linear(288, 270) → LayerNorm → GELU → output (270)
    """

    OUTPUT_DIM = 270

    def __init__(self):
        super().__init__()
        self.raw_encoder = QuaternionStateEncoder()

        # Projection network: 288 raw → 270 output
        self.projection = nn.Sequential(
            nn.Linear(QuaternionStateEncoder.RAW_DIM, 270),
            nn.LayerNorm(270),
            nn.GELU(),
        )

        # Geometric feature integration
        self.geo_projection = nn.Sequential(
            nn.Linear(20, 64),
            nn.GELU(),
            nn.Linear(64, 270),
        )

        # Combine main encoding + geometric features
        self.combine = nn.Sequential(
            nn.Linear(270 * 2, 270),
            nn.LayerNorm(270),
            nn.GELU(),
        )

    def encode_state(self, state: dict) -> np.ndarray:
        """Encode a single state to raw numpy vector (no gradient)."""
        return self.raw_encoder.encode(state)

    def encode_state_batch(self, states: List[dict]) -> np.ndarray:
        """Encode batch of states to raw numpy (no gradient)."""
        return self.raw_encoder.encode_batch(states)

    def forward(self, raw_encoding: torch.Tensor,
                geo_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Project raw encoding to 270-dim output.

        Args:
            raw_encoding: (batch, 288) tensor of raw quaternion encodings
            geo_features: (batch, 20) optional geometric features

        Returns:
            (batch, 270) projected state vector
        """
        main = self.projection(raw_encoding)

        if geo_features is not None:
            geo = self.geo_projection(geo_features)
            combined = torch.cat([main, geo], dim=-1)
            return self.combine(combined)

        return main

    def encode_and_project(self, states: List[dict],
                           device: torch.device) -> torch.Tensor:
        """
        Full pipeline: raw state dicts → 270-dim tensor.
        Convenience method for training/inference.
        """
        # Raw encoding
        raw = self.raw_encoder.encode_batch(states)
        raw_tensor = torch.FloatTensor(raw).to(device)

        # Geometric features
        geo_list = [self.raw_encoder.compute_geometric_features(s) for s in states]
        geo_tensor = torch.FloatTensor(np.stack(geo_list)).to(device)

        return self.forward(raw_tensor, geo_tensor)
