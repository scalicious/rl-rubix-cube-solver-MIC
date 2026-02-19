"""
agents.py - Multi-agent actor-critic networks with ICM curiosity.

Three specialized PPO agents (corner, edge, center), each with their own
actor-critic and curiosity module. They all receive the same 270-dim state
but learn different internal representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple
import numpy as np


class ResidualBlock(nn.Module):
    """Pre-activation residual block with layer norm."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ActorCritic(nn.Module):
    """
    Shared-backbone actor-critic with residual connections.
    Input (270) -> backbone (512 -> 2x ResBlock -> 256) -> actor (18) / critic (1)
    """

    def __init__(self, state_dim=270, action_dim=18, hidden_dim=512):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # smaller init for output heads
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, state):
        features = self.backbone(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_policy(self, state):
        """Returns action probabilities."""
        features = self.backbone(state)
        logits = self.actor(features)
        return F.softmax(logits, dim=-1)

    def get_action(self, state, deterministic=False):
        """Sample an action. Returns (action, log_prob, value)."""
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze(-1)

    def evaluate_actions(self, states, actions):
        """Evaluate log_probs, values, and entropy for given state-action pairs (PPO update)."""
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy


class CuriosityModule(nn.Module):
    """
    ICM curiosity — forward model predicts next state embedding,
    inverse model predicts action from (state, next_state).
    Curiosity reward = forward prediction error.
    """

    def __init__(self, state_dim=270, action_dim=18, feature_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.feature_dim = feature_dim

        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.GELU(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # predicts next state embedding from (state_embed, action_one_hot)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, feature_dim),
        )

        # predicts action from (state_embed, next_state_embed)
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state, next_state, action):
        # encode both states
        phi_s = self.feature_encoder(state)
        phi_s_next = self.feature_encoder(next_state)

        # forward model
        action_one_hot = F.one_hot(action.long(), self.action_dim).float()
        forward_input = torch.cat([phi_s, action_one_hot], dim=-1)
        pred_phi_s_next = self.forward_model(forward_input)

        # curiosity = prediction error
        curiosity_reward = 0.5 * ((pred_phi_s_next - phi_s_next.detach()) ** 2).sum(dim=-1)

        forward_loss = F.mse_loss(pred_phi_s_next, phi_s_next.detach())

        # inverse model
        inverse_input = torch.cat([phi_s, phi_s_next], dim=-1)
        pred_action_logits = self.inverse_model(inverse_input)
        inverse_loss = F.cross_entropy(pred_action_logits, action.long())

        return curiosity_reward, forward_loss, inverse_loss


class BaseAgent(nn.Module):
    """Base agent = ActorCritic + CuriosityModule."""

    def __init__(self, state_dim=270, action_dim=18, agent_name="base"):
        super().__init__()
        self.agent_name = agent_name
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.curiosity = CuriosityModule(state_dim, action_dim)

    def get_policy(self, state):
        return self.actor_critic.get_policy(state)

    def get_action(self, state, deterministic=False):
        return self.actor_critic.get_action(state, deterministic)

    def evaluate_actions(self, states, actions):
        return self.actor_critic.evaluate_actions(states, actions)

    def compute_curiosity(self, state, next_state, action):
        return self.curiosity(state, next_state, action)

    def get_value(self, state):
        _, value = self.actor_critic(state)
        return value.squeeze(-1)


class CornerAgent(BaseAgent):
    """Handles the 8 corner cubies (3 orientations each, 8! permutations)."""
    def __init__(self, state_dim=270, action_dim=18):
        super().__init__(state_dim, action_dim, agent_name="corner")


class EdgeAgent(BaseAgent):
    """Handles the 12 edge cubies (2 orientations, 12! permutations)."""
    def __init__(self, state_dim=270, action_dim=18):
        super().__init__(state_dim, action_dim, agent_name="edge")


class CenterAgent(BaseAgent):
    """Handles center/face orientation alignment."""
    def __init__(self, state_dim=270, action_dim=18):
        super().__init__(state_dim, action_dim, agent_name="center")
