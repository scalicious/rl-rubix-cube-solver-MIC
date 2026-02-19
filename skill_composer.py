"""
skill_composer.py - Attention-based policy blending.

Dynamically weights the 3 agent policies based on current cube state.
Uses scaled dot-product attention + a small residual policy refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np


class SkillComposer(nn.Module):
    """
    Takes encoded state + individual agent policies, produces a
    weighted blend as the final policy. Has its own value head
    (meta-critic) for the composed policy.
    """

    def __init__(self, state_dim=270, num_agents=3, embed_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.embed_dim = embed_dim

        # state -> query for attention
        self.query_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # learnable key per agent
        self.agent_keys = nn.Parameter(
            torch.randn(num_agents, embed_dim) * 0.1
        )

        # learnable temperature
        self.temperature = nn.Parameter(torch.ones(1) * np.sqrt(embed_dim))

        # residual correction on top of blended policy
        self.policy_refine = nn.Sequential(
            nn.Linear(state_dim + 18, 128),
            nn.GELU(),
            nn.Linear(128, 18),
        )
        # start with near-zero refinement
        nn.init.zeros_(self.policy_refine[-1].weight)
        nn.init.zeros_(self.policy_refine[-1].bias)

        # value head for composed policy
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.query_net, self.value_head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)

    def compute_weights(self, state):
        """Attention weights over agents, shape (batch, num_agents)."""
        query = self.query_net(state)
        scores = torch.matmul(query, self.agent_keys.T)
        scores = scores / self.temperature.clamp(min=1.0)
        return F.softmax(scores, dim=-1)

    def forward(self, state, agent_policies):
        """
        Blend agent policies with learned attention weights,
        then apply a small residual refinement.
        """
        weights = self.compute_weights(state)  # (batch, num_agents)

        # weighted sum of agent policies
        policy_stack = torch.stack(agent_policies, dim=1)  # (batch, num_agents, 18)
        blended = torch.bmm(
            weights.unsqueeze(1),
            policy_stack
        ).squeeze(1)  # (batch, 18)

        # residual refinement
        refine_input = torch.cat([state, blended], dim=-1)
        refinement = self.policy_refine(refine_input)

        # apply as logit adjustment, re-normalize
        refined_logits = torch.log(blended + 1e-8) + 0.1 * refinement
        final_policy = F.softmax(refined_logits, dim=-1)

        return final_policy

    def get_value(self, state):
        return self.value_head(state).squeeze(-1)

    def get_action(self, state, agent_policies, deterministic=False):
        """Sample action from composed policy. Returns (action, log_prob, value)."""
        final_policy = self.forward(state, agent_policies)
        value = self.get_value(state)

        dist = torch.distributions.Categorical(final_policy)

        if deterministic:
            action = torch.argmax(final_policy, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def evaluate_composed(self, state, agent_policies, actions):
        """For PPO update: returns (log_probs, values, entropy)."""
        final_policy = self.forward(state, agent_policies)
        values = self.get_value(state)

        dist = torch.distributions.Categorical(final_policy)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy
