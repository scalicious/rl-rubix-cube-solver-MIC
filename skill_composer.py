"""
skill_composer.py — Attention-Based Policy Blending for MARVELS
================================================================
The Skill Composer dynamically weights the policies of the three
specialized agents (Corner, Edge, Center) based on the current
cube state, producing a single blended policy for action selection.

Architecture:
  state (270-d) → MLP → query vector
  agent_embeddings (3 learnable) → keys
  attention_scores = softmax(query · keys^T / √d)
  final_policy = Σ scores_i × agent_i_policy

Novel Aspect:
  Unlike simple ensemble averaging, the Skill Composer learns
  *when* to prioritize each agent. Early in solving, it weights
  the CornerAgent higher; later, it shifts focus to edges and
  centers. This mimics the human CFOP strategy but is learned
  entirely from data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np


class SkillComposer(nn.Module):
    """
    Attention-based skill composition network.

    Takes the encoded cube state and the individual agent policies,
    then produces a weighted combination as the final policy.

    Also maintains its own value head for the composed policy,
    which serves as a "meta-critic" for the overall solve progress.
    """

    def __init__(self, state_dim: int = 270, num_agents: int = 3,
                 embed_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.embed_dim = embed_dim

        # ──── Query Network ────
        # Transforms state into a query vector for attention
        self.query_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ──── Agent Key Embeddings ────
        # Each agent has a learnable key vector
        self.agent_keys = nn.Parameter(
            torch.randn(num_agents, embed_dim) * 0.1
        )

        # ──── Temperature for attention (learnable) ────
        # Controls sharpness of attention distribution
        self.temperature = nn.Parameter(torch.ones(1) * np.sqrt(embed_dim))

        # ──── Policy Refinement ────
        # Optional residual correction applied to the blended policy
        self.policy_refine = nn.Sequential(
            nn.Linear(state_dim + 18, 128),
            nn.GELU(),
            nn.Linear(128, 18),
        )
        # Small initial weight so refinement starts near zero
        nn.init.zeros_(self.policy_refine[-1].weight)
        nn.init.zeros_(self.policy_refine[-1].bias)

        # ──── Meta Value Head ────
        # Value estimate for the composed policy
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

    def compute_weights(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights for each agent.

        Args:
            state: (batch, 270) encoded state

        Returns:
            weights: (batch, num_agents) attention weights summing to 1
        """
        # Query from state
        query = self.query_net(state)  # (batch, embed_dim)

        # Scaled dot-product attention
        # query: (batch, embed_dim), keys: (num_agents, embed_dim)
        scores = torch.matmul(query, self.agent_keys.T)  # (batch, num_agents)
        scores = scores / self.temperature.clamp(min=1.0)

        weights = F.softmax(scores, dim=-1)  # (batch, num_agents)
        return weights

    def forward(self, state: torch.Tensor,
                agent_policies: List[torch.Tensor]) -> torch.Tensor:
        """
        Compose agent policies into a single blended policy.

        Args:
            state: (batch, 270) encoded state
            agent_policies: list of 3 tensors, each (batch, 18) probability distributions

        Returns:
            final_policy: (batch, 18) blended and refined probability distribution
        """
        # Compute attention weights
        weights = self.compute_weights(state)  # (batch, num_agents)

        # Stack agent policies: (batch, num_agents, 18)
        policy_stack = torch.stack(agent_policies, dim=1)

        # Weighted sum: (batch, 18)
        blended = torch.bmm(
            weights.unsqueeze(1),     # (batch, 1, num_agents)
            policy_stack               # (batch, num_agents, 18)
        ).squeeze(1)                   # (batch, 18)

        # Residual policy refinement
        refine_input = torch.cat([state, blended], dim=-1)
        refinement = self.policy_refine(refine_input)

        # Apply refinement as logit adjustment, then re-normalize
        refined_logits = torch.log(blended + 1e-8) + 0.1 * refinement
        final_policy = F.softmax(refined_logits, dim=-1)

        return final_policy

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Meta-value estimate for the composed policy.

        Args:
            state: (batch, 270)

        Returns:
            value: (batch,) scalar values
        """
        return self.value_head(state).squeeze(-1)

    def get_action(self, state: torch.Tensor,
                   agent_policies: List[torch.Tensor],
                   deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from composed policy.

        Returns:
            action: int
            log_prob: log probability of selected action
            value: meta-value estimate
        """
        final_policy = self.forward(state, agent_policies)
        value = self.get_value(state)

        dist = torch.distributions.Categorical(final_policy)

        if deterministic:
            action = torch.argmax(final_policy, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def evaluate_composed(self, state: torch.Tensor,
                          agent_policies: List[torch.Tensor],
                          actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the composed policy for PPO update.

        Returns:
            log_probs: (batch,) log probabilities
            values: (batch,) state values
            entropy: (batch,) policy entropy
        """
        final_policy = self.forward(state, agent_policies)
        values = self.get_value(state)

        dist = torch.distributions.Categorical(final_policy)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy
