"""
agents.py — Multi-Agent Actor-Critic Networks with Curiosity for MARVELS
=========================================================================
Implements three specialized PPO actor-critic agents:
  1. CornerAgent  — focuses on solving the 8 corners
  2. EdgeAgent    — focuses on solving the 12 edges
  3. CenterAgent  — focuses on center/orientation alignment

Each agent has:
  - Actor:  state → action logits (18 actions)
  - Critic: state → scalar value
  - ICM (Intrinsic Curiosity Module):
      • Forward model:  (state_embed, action) → predicted next_state_embed
      • Inverse model:  (state_embed, next_state_embed) → predicted action
      • Curiosity reward = MSE of forward model prediction error

Novel Aspect:
  Each agent receives the same 270-dim state but learns a *different
  internal representation* optimized for its cubie group (corners, edges,
  or centers). The Skill Composer then blends their policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Dict, Optional
import numpy as np


# ──────────────────────── Residual Block ────────────────────────

class ResidualBlock(nn.Module):
    """
    Pre-activation residual block with layer normalization.
    Enables deeper networks without degradation — critical for
    learning the complex combinatorial structure of the cube.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ──────────────────── Actor-Critic Network ────────────────────

class ActorCritic(nn.Module):
    """
    Shared-backbone actor-critic with residual connections.

    Architecture:
      Input (270) → Linear(512) → [ResBlock × 2] → 256-dim backbone
      Backbone → Actor head  → 18 logits
      Backbone → Critic head → 1 value
    """

    def __init__(self, state_dim: int = 270, action_dim: int = 18,
                 hidden_dim: int = 512):
        super().__init__()

        # Shared backbone with residual connections
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # Actor head: outputs action logits
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
        )

        # Critic head: outputs state value
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Initialize weights with smaller scale for stability
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small initialization for policy head (more uniform initial policy)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        # Small initialization for value head
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, 270) encoded state

        Returns:
            logits: (batch, 18) action logits
            value:  (batch, 1) state value
        """
        features = self.backbone(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_policy(self, state: torch.Tensor) -> torch.Tensor:
        """Return action probabilities (softmax of logits)."""
        features = self.backbone(state)
        logits = self.actor(features)
        return F.softmax(logits, dim=-1)

    def get_action(self, state: torch.Tensor,
                   deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Returns:
            action: int, selected action index
            log_prob: log probability of the selected action
            value: state value estimate
        """
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze(-1)

    def evaluate_actions(self, states: torch.Tensor,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities, values, and entropy for given state-action pairs.
        Used during PPO update.
        """
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy


# ──────────────── Intrinsic Curiosity Module (ICM) ────────────────

class CuriosityModule(nn.Module):
    """
    ICM-style curiosity module that provides intrinsic reward
    based on prediction error of a forward dynamics model.

    Components:
      1. Feature encoder: maps state to a compact embedding
      2. Forward model: predicts next state embedding from (state, action)
      3. Inverse model: predicts action from (state, next_state)

    The curiosity reward is the MSE between predicted and actual
    next-state embeddings. High error → novel state → high reward.

    Novel Aspect:
      By providing curiosity reward *per agent*, each agent explores
      the state space relevant to its cubie group independently.
    """

    def __init__(self, state_dim: int = 270, action_dim: int = 18,
                 feature_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.feature_dim = feature_dim

        # State feature encoder (shared by forward and inverse models)
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.GELU(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # Forward model: (feature, action_one_hot) → predicted_next_feature
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, feature_dim),
        )

        # Inverse model: (feature, next_feature) → predicted_action_logits
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor, next_state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute curiosity signals.

        Args:
            state:      (batch, 270)
            next_state: (batch, 270)
            action:     (batch,) action indices

        Returns:
            curiosity_reward: (batch,) intrinsic rewards
            forward_loss:     scalar, MSE of forward model
            inverse_loss:     scalar, CE of inverse model
        """
        # Encode states
        phi_s = self.feature_encoder(state)
        phi_s_next = self.feature_encoder(next_state)

        # Forward model prediction
        action_one_hot = F.one_hot(action.long(), self.action_dim).float()
        forward_input = torch.cat([phi_s, action_one_hot], dim=-1)
        pred_phi_s_next = self.forward_model(forward_input)

        # Curiosity reward = prediction error (per-sample)
        curiosity_reward = 0.5 * ((pred_phi_s_next - phi_s_next.detach()) ** 2).sum(dim=-1)

        # Forward loss (for training the forward model)
        forward_loss = F.mse_loss(pred_phi_s_next, phi_s_next.detach())

        # Inverse model prediction
        inverse_input = torch.cat([phi_s, phi_s_next], dim=-1)
        pred_action_logits = self.inverse_model(inverse_input)
        inverse_loss = F.cross_entropy(pred_action_logits, action.long())

        return curiosity_reward, forward_loss, inverse_loss


# ──────────────── Specialized Agent Classes ────────────────

class BaseAgent(nn.Module):
    """
    Base agent combining ActorCritic + CuriosityModule.
    Subclasses specialize by name and can override reward shaping.
    """

    def __init__(self, state_dim: int = 270, action_dim: int = 18,
                 agent_name: str = "base"):
        super().__init__()
        self.agent_name = agent_name
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.curiosity = CuriosityModule(state_dim, action_dim)

    def get_policy(self, state: torch.Tensor) -> torch.Tensor:
        """Return action probabilities."""
        return self.actor_critic.get_policy(state)

    def get_action(self, state: torch.Tensor,
                   deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        return self.actor_critic.get_action(state, deterministic)

    def evaluate_actions(self, states: torch.Tensor,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log_probs, values, entropy."""
        return self.actor_critic.evaluate_actions(states, actions)

    def compute_curiosity(self, state: torch.Tensor, next_state: torch.Tensor,
                          action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute curiosity reward and losses."""
        return self.curiosity(state, next_state, action)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        _, value = self.actor_critic(state)
        return value.squeeze(-1)


class CornerAgent(BaseAgent):
    """
    Specialized agent for solving the 8 corner cubies.

    Corners have 3 possible orientations each and 8! possible
    permutations, making them the most constrained pieces.
    This agent learns to prioritize corner-solving moves.
    """

    def __init__(self, state_dim: int = 270, action_dim: int = 18):
        super().__init__(state_dim, action_dim, agent_name="corner")


class EdgeAgent(BaseAgent):
    """
    Specialized agent for solving the 12 edge cubies.

    Edges have 2 possible orientations and 12! permutations.
    This agent learns edge-pairing and insertion moves.
    """

    def __init__(self, state_dim: int = 270, action_dim: int = 18):
        super().__init__(state_dim, action_dim, agent_name="edge")


class CenterAgent(BaseAgent):
    """
    Specialized agent for center/orientation alignment.

    On a standard 3×3 cube, centers are fixed. This agent
    focuses on aligning surrounding stickers with their
    center colors — effectively learning whole-face orientation.
    """

    def __init__(self, state_dim: int = 270, action_dim: int = 18):
        super().__init__(state_dim, action_dim, agent_name="center")
