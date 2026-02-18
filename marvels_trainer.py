"""
marvels_trainer.py — Full PPO Training Loop for MARVELS
========================================================
Orchestrates the multi-agent training pipeline:

  1. Collect rollouts from vectorized environments
  2. Compute rewards = external + 0.5 × curiosity
  3. Calculate GAE advantages
  4. PPO update for each agent + skill composer
  5. Curriculum: increase scramble depth on success

Training Flow (per iteration):
  ┌─────────────────────────────────────────────────┐
  │  For each env step:                              │
  │    state → encoder → 270-d                       │
  │    270-d → each agent → 3 policies               │
  │    3 policies + state → composer → final policy   │
  │    sample action → env.step()                    │
  │    compute curiosity rewards                     │
  │    store transition                               │
  │  End rollout                                     │
  │                                                  │
  │  Compute GAE advantages                          │
  │  PPO update (clip=0.2, entropy=0.01, epochs=4)   │
  │  Update curriculum                               │
  └─────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from collections import deque

from rubiks_env import RubiksCube, VectorizedRubiksCube, NUM_ACTIONS, ACTION_NAMES
from quaternion_encoder import QuaternionEncoder
from agents import CornerAgent, EdgeAgent, CenterAgent, BaseAgent
from skill_composer import SkillComposer
from utils import (
    MovingAverage, save_checkpoint, load_checkpoint,
    format_time, visualize_cube_compact
)


# ──────────────────── Rollout Buffer ────────────────────

class RolloutBuffer:
    """
    Stores transitions from rollout collection for PPO updates.

    Stores per-step:
      - states (encoded), actions, log_probs, rewards, values, dones
      - curiosity rewards per agent
      - Agent-specific data for individual agent updates
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.curiosity_rewards = []

        # Per-agent data
        self.agent_log_probs = [[], [], []]
        self.agent_values = [[], [], []]

        # Next states for curiosity computation
        self.next_states = []

    def add(self, state, action, log_prob, reward, value, done,
            curiosity_reward, agent_log_probs, agent_values, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.curiosity_rewards.append(curiosity_reward)
        self.next_states.append(next_state)

        for i in range(3):
            self.agent_log_probs[i].append(agent_log_probs[i])
            self.agent_values[i].append(agent_values[i])

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


# ──────────────── GAE Computation ────────────────

def compute_gae(rewards: List[float], values: List[float],
                dones: List[bool], next_value: float,
                gamma: float = 0.999, lam: float = 0.95) -> Tuple[List[float], List[float]]:
    """
    Generalized Advantage Estimation (GAE-λ).

    Args:
        rewards: list of rewards per step
        values: list of value estimates per step
        dones: list of done flags per step
        next_value: value estimate of the state after the last step
        gamma: discount factor (high because cube reward is sparse)
        lam: GAE lambda (bias-variance tradeoff)

    Returns:
        advantages: list of GAE advantages
        returns: list of discounted returns (advantages + values)
    """
    advantages = []
    gae = 0.0

    # Work backwards through the trajectory
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        # Mask: if done, next value is 0
        mask = 0.0 if dones[t] else 1.0

        # TD error
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


# ──────────────── MARVELS Trainer ────────────────

class MARVELSTrainer:
    """
    Main training class for the MARVELS algorithm.

    Manages:
      - 3 specialized agents (corner, edge, center)
      - Skill composer (attention-based blending)
      - Quaternion encoder
      - PPO optimization
      - Curriculum learning
      - Checkpointing and logging
    """

    def __init__(self, config: Optional[Dict] = None):
        # ──── Configuration ────
        self.config = {
            'num_envs': 16,               # Parallel environments
            'rollout_length': 128,         # Steps per rollout
            'ppo_epochs': 4,               # PPO update epochs
            'mini_batch_size': 64,         # Mini-batch size
            'clip_ratio': 0.2,             # PPO clip parameter
            'entropy_coef': 0.01,          # Entropy bonus coefficient
            'value_coef': 0.5,             # Value loss coefficient
            'curiosity_coef': 0.5,         # Curiosity reward scaling
            'max_grad_norm': 0.5,          # Gradient clipping
            'lr': 3e-4,                    # Learning rate
            'gamma': 0.999,               # Discount factor
            'gae_lambda': 0.95,           # GAE lambda
            'curiosity_lr': 1e-3,         # Curiosity module LR
            'initial_scramble_depth': 1,  # Curriculum start
            'max_scramble_depth': 25,     # Curriculum max
            'curriculum_threshold': 0.5,  # Success rate to advance
            'max_moves': 200,             # Episode max length
            'device': 'cpu',
            'save_dir': './checkpoints',
        }
        if config:
            self.config.update(config)

        self.device = torch.device(self.config['device'])

        # ──── Initialize Components ────
        self.encoder = QuaternionEncoder().to(self.device)
        self.agents = [
            CornerAgent(270, NUM_ACTIONS).to(self.device),
            EdgeAgent(270, NUM_ACTIONS).to(self.device),
            CenterAgent(270, NUM_ACTIONS).to(self.device),
        ]
        self.composer = SkillComposer(270, 3).to(self.device)

        # ──── Optimizers ────
        # Single optimizer for all components (joint training)
        all_params = []
        all_params += list(self.encoder.parameters())
        for agent in self.agents:
            all_params += list(agent.parameters())
        all_params += list(self.composer.parameters())

        self.optimizer = optim.Adam(all_params, lr=self.config['lr'], eps=1e-5)

        # Separate optimizer for curiosity modules (different learning rate)
        curiosity_params = []
        for agent in self.agents:
            curiosity_params += list(agent.curiosity.parameters())
        self.curiosity_optimizer = optim.Adam(
            curiosity_params, lr=self.config['curiosity_lr'], eps=1e-5
        )

        # ──── Environments ────
        self.vec_env = VectorizedRubiksCube(
            self.config['num_envs'],
            max_moves=self.config['max_moves']
        )

        # ──── Tracking ────
        self.current_scramble_depth = self.config['initial_scramble_depth']
        self.total_steps = 0
        self.total_episodes = 0
        self.best_solve_rate = 0.0
        self.solve_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        self.move_count_history = deque(maxlen=100)

    def _encode_states(self, states: List[dict]) -> torch.Tensor:
        """Encode a list of state dicts into a batch tensor."""
        return self.encoder.encode_and_project(states, self.device)

    # ──────────────── Rollout Collection ────────────────

    def collect_rollout(self) -> RolloutBuffer:
        """
        Collect a full rollout from all parallel environments.

        For each step:
          1. Encode state via quaternion encoder
          2. Get individual agent policies
          3. Compose final policy via skill composer
          4. Sample action, step environment
          5. Compute curiosity rewards
          6. Store transition

        Returns filled RolloutBuffer.
        """
        buffer = RolloutBuffer()

        # Reset all envs with current scramble depth
        states = self.vec_env.reset_all(self.current_scramble_depth)

        for step in range(self.config['rollout_length']):
            with torch.no_grad():
                # Encode current states
                state_tensor = self._encode_states(states)

                # Get individual agent policies and values
                agent_policies = []
                agent_log_probs_list = []
                agent_values_list = []

                for agent in self.agents:
                    policy = agent.get_policy(state_tensor)
                    _, value = agent.actor_critic(state_tensor)
                    agent_policies.append(policy)
                    agent_values_list.append(value.squeeze(-1))

                # Compose final policy
                final_policy = self.composer(state_tensor, agent_policies)
                meta_value = self.composer.get_value(state_tensor)

                # Sample actions (one per env)
                dist = torch.distributions.Categorical(final_policy)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            # Step all environments
            actions_list = actions.cpu().numpy().tolist()
            next_states, rewards, dones, infos = self.vec_env.step(actions_list)

            # Compute curiosity rewards
            with torch.no_grad():
                next_state_tensor = self._encode_states(next_states)
                total_curiosity = torch.zeros(self.config['num_envs'], device=self.device)

                for agent in self.agents:
                    curiosity_r, _, _ = agent.compute_curiosity(
                        state_tensor, next_state_tensor, actions
                    )
                    total_curiosity += curiosity_r

                # Average curiosity across agents
                avg_curiosity = total_curiosity / len(self.agents)

            # Compute total reward per env
            for env_idx in range(self.config['num_envs']):
                ext_reward = rewards[env_idx]
                cur_reward = avg_curiosity[env_idx].item()
                total_reward = ext_reward + self.config['curiosity_coef'] * cur_reward

                # Store per-agent log probs and values
                per_agent_lp = [alp[env_idx].item() for alp in agent_log_probs_list] if agent_log_probs_list else [0.0, 0.0, 0.0]
                per_agent_v = [av[env_idx].item() for av in agent_values_list]

                buffer.add(
                    state=state_tensor[env_idx].cpu().numpy(),
                    action=actions[env_idx].item(),
                    log_prob=log_probs[env_idx].item(),
                    reward=total_reward,
                    value=meta_value[env_idx].item(),
                    done=dones[env_idx],
                    curiosity_reward=cur_reward,
                    agent_log_probs=per_agent_lp,
                    agent_values=per_agent_v,
                    next_state=next_state_tensor[env_idx].cpu().numpy(),
                )

                # Track episode completion
                if dones[env_idx]:
                    self.total_episodes += 1
                    self.solve_history.append(infos[env_idx]['solved'])
                    self.reward_history.append(ext_reward)
                    self.move_count_history.append(infos[env_idx]['move_count'])

            # Reset done environments
            for env_idx in range(self.config['num_envs']):
                if dones[env_idx]:
                    next_states[env_idx] = self.vec_env.reset_one(
                        env_idx, self.current_scramble_depth
                    )

            states = next_states
            self.total_steps += self.config['num_envs']

        return buffer

    # ──────────────── PPO Update ────────────────

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Perform PPO update on all components using collected rollout data.

        Steps:
          1. Compute GAE advantages
          2. For each PPO epoch, shuffle data into mini-batches
          3. Compute policy loss (clipped surrogate)
          4. Compute value loss
          5. Compute entropy bonus
          6. Compute curiosity losses (forward + inverse)
          7. Backprop through all components jointly

        Returns dict of loss metrics.
        """
        # ──── Prepare data ────
        states_np = np.array(buffer.states)
        next_states_np = np.array(buffer.next_states)
        actions_np = np.array(buffer.actions)
        old_log_probs_np = np.array(buffer.log_probs)
        rewards_list = buffer.rewards
        values_list = buffer.values
        dones_list = buffer.dones

        # Compute last value for GAE
        with torch.no_grad():
            last_state = torch.FloatTensor(states_np[-1:]).to(self.device)
            last_value = self.composer.get_value(last_state).item()

        # Compute GAE
        advantages, returns = compute_gae(
            rewards_list, values_list, dones_list, last_value,
            gamma=self.config['gamma'], lam=self.config['gae_lambda']
        )

        # Convert to tensors
        states_t = torch.FloatTensor(states_np).to(self.device)
        next_states_t = torch.FloatTensor(next_states_np).to(self.device)
        actions_t = torch.LongTensor(actions_np).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs_np).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # ──── PPO Epochs ────
        num_samples = len(buffer)
        batch_size = self.config['mini_batch_size']
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'curiosity_loss': 0.0,
            'total_loss': 0.0,
        }
        num_updates = 0

        for epoch in range(self.config['ppo_epochs']):
            # Shuffle indices
            indices = np.random.permutation(num_samples)

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = indices[start:end]

                b_states = states_t[batch_idx]
                b_next_states = next_states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_advantages = advantages_t[batch_idx]
                b_returns = returns_t[batch_idx]

                # ──── Forward pass through all agents ────
                agent_policies = []
                agent_entropies = []

                for agent in self.agents:
                    policy = agent.get_policy(b_states)
                    agent_policies.append(policy)

                    # Agent entropy for exploration
                    dist = torch.distributions.Categorical(policy)
                    agent_entropies.append(dist.entropy().mean())

                # ──── Forward pass through composer ────
                log_probs, values, entropy = self.composer.evaluate_composed(
                    b_states, agent_policies, b_actions
                )

                # ──── Policy loss (clipped surrogate) ────
                ratio = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.config['clip_ratio'],
                                    1 + self.config['clip_ratio']) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # ──── Value loss ────
                value_loss = F.mse_loss(values, b_returns)

                # ──── Entropy bonus ────
                total_entropy = entropy.mean()
                for ae in agent_entropies:
                    total_entropy += ae * 0.1  # Smaller weight for individual agents

                # ──── Curiosity losses ────
                total_forward_loss = torch.tensor(0.0, device=self.device)
                total_inverse_loss = torch.tensor(0.0, device=self.device)

                for agent in self.agents:
                    cur_reward, fwd_loss, inv_loss = agent.compute_curiosity(
                        b_states, b_next_states, b_actions
                    )
                    total_forward_loss += fwd_loss
                    total_inverse_loss += inv_loss

                curiosity_loss = (total_forward_loss + total_inverse_loss) / len(self.agents)

                # ──── Total loss ────
                total_loss = (
                    policy_loss
                    + self.config['value_coef'] * value_loss
                    - self.config['entropy_coef'] * total_entropy
                )

                # ──── Update main parameters ────
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self._all_params(), self.config['max_grad_norm']
                )
                self.optimizer.step()

                # ──── Update curiosity modules separately ────
                self.curiosity_optimizer.zero_grad()
                curiosity_loss.backward()
                self.curiosity_optimizer.step()

                # Track metrics
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += total_entropy.item()
                metrics['curiosity_loss'] += curiosity_loss.item()
                metrics['total_loss'] += total_loss.item()
                num_updates += 1

        # Average metrics
        for k in metrics:
            metrics[k] /= max(num_updates, 1)

        return metrics

    def _all_params(self):
        """Get all trainable parameters for gradient clipping."""
        params = list(self.encoder.parameters())
        for agent in self.agents:
            params += list(agent.actor_critic.parameters())
        params += list(self.composer.parameters())
        return params

    # ──────────────── Curriculum ────────────────

    def update_curriculum(self):
        """
        Increase scramble depth when solve rate exceeds threshold.
        This implements curriculum learning: start easy, get harder.
        """
        if len(self.solve_history) < 20:
            return

        solve_rate = sum(self.solve_history) / len(self.solve_history)

        if (solve_rate >= self.config['curriculum_threshold'] and
                self.current_scramble_depth < self.config['max_scramble_depth']):
            self.current_scramble_depth += 1
            self.solve_history.clear()
            print(f"  📈 Curriculum advanced → scramble depth = {self.current_scramble_depth}")

    # ──────────────── Training Loop ────────────────

    def train(self, num_iterations: int = 1000,
              log_interval: int = 10,
              save_interval: int = 50) -> Dict:
        """
        Main training loop.

        Args:
            num_iterations: number of rollout + update cycles
            log_interval: print stats every N iterations
            save_interval: save checkpoint every N iterations

        Returns:
            dict with training history
        """
        print("=" * 60)
        print("  MARVELS Training")
        print("  Multi-Agent Residual Vision-Enhanced Learning")
        print("  for Symbolic Reasoning")
        print("=" * 60)
        print(f"  Device:          {self.device}")
        print(f"  Num envs:        {self.config['num_envs']}")
        print(f"  Rollout length:  {self.config['rollout_length']}")
        print(f"  PPO epochs:      {self.config['ppo_epochs']}")
        print(f"  Initial scramble: {self.current_scramble_depth}")
        print("=" * 60)
        print()

        history = {
            'solve_rates': [],
            'avg_rewards': [],
            'avg_moves': [],
            'losses': [],
            'scramble_depths': [],
        }

        start_time = time.time()

        for iteration in range(1, num_iterations + 1):
            iter_start = time.time()

            # ──── Collect rollout ────
            buffer = self.collect_rollout()

            # ──── PPO update ────
            metrics = self.update(buffer)

            # ──── Update curriculum ────
            self.update_curriculum()

            # ──── Logging ────
            solve_rate = (sum(self.solve_history) / len(self.solve_history)
                          if self.solve_history else 0.0)
            avg_reward = (sum(self.reward_history) / len(self.reward_history)
                          if self.reward_history else 0.0)
            avg_moves = (sum(self.move_count_history) / len(self.move_count_history)
                         if self.move_count_history else 0.0)

            history['solve_rates'].append(solve_rate)
            history['avg_rewards'].append(avg_reward)
            history['avg_moves'].append(avg_moves)
            history['losses'].append(metrics)
            history['scramble_depths'].append(self.current_scramble_depth)

            if iteration % log_interval == 0:
                elapsed = time.time() - start_time
                iter_time = time.time() - iter_start
                steps_per_sec = self.total_steps / elapsed

                print(f"  Iter {iteration:5d}/{num_iterations} │ "
                      f"Scramble: {self.current_scramble_depth:2d} │ "
                      f"Solve: {solve_rate:.1%} │ "
                      f"Reward: {avg_reward:+7.2f} │ "
                      f"Moves: {avg_moves:5.1f} │ "
                      f"Loss: {metrics['total_loss']:.4f} │ "
                      f"Curiosity: {metrics['curiosity_loss']:.4f} │ "
                      f"Entropy: {metrics['entropy']:.3f} │ "
                      f"{steps_per_sec:.0f} steps/s │ "
                      f"{format_time(elapsed)}")

            # ──── Save checkpoint ────
            if iteration % save_interval == 0:
                self._save(f"checkpoint_iter_{iteration}")

            # Save best model
            if solve_rate > self.best_solve_rate and len(self.solve_history) >= 20:
                self.best_solve_rate = solve_rate
                self._save("best_model")
                print(f"  ⭐ New best solve rate: {solve_rate:.1%}")

        total_time = time.time() - start_time
        print()
        print("=" * 60)
        print(f"  Training complete in {format_time(total_time)}")
        print(f"  Total steps: {self.total_steps:,}")
        print(f"  Total episodes: {self.total_episodes:,}")
        print(f"  Best solve rate: {self.best_solve_rate:.1%}")
        print(f"  Final scramble depth: {self.current_scramble_depth}")
        print("=" * 60)

        return history

    # ──────────────── Solve Demo ────────────────

    def solve(self, scramble_depth: int = 15,
              max_moves: int = 200,
              verbose: bool = True) -> Tuple[bool, int, List[str]]:
        """
        Attempt to solve a scrambled cube using the trained policy.
        No search — pure policy inference.

        Args:
            scramble_depth: number of random moves to scramble
            max_moves: maximum solve attempts
            verbose: print step-by-step progress

        Returns:
            (solved, num_moves, move_list)
        """
        env = RubiksCube()
        scramble_actions = env.scramble(scramble_depth)

        if verbose:
            print(f"\n  Scrambled with {scramble_depth} moves: "
                  f"{[ACTION_NAMES[a] for a in scramble_actions]}")
            print(f"\n  Initial state:")
            print(env.render())
            print()

        move_list = []

        for step in range(max_moves):
            state = env.get_state()

            with torch.no_grad():
                state_tensor = self._encode_states([state])

                # Get agent policies
                agent_policies = [
                    agent.get_policy(state_tensor)
                    for agent in self.agents
                ]

                # Compose and get weights
                weights = self.composer.compute_weights(state_tensor)
                final_policy = self.composer(state_tensor, agent_policies)

                # Deterministic action selection
                action = torch.argmax(final_policy, dim=-1).item()

            _, reward, done, info = env.step(action)
            move_list.append(ACTION_NAMES[action])

            if verbose and step < 50:  # Print first 50 moves
                w = weights[0].cpu().numpy()
                print(f"    Step {step+1:3d}: {ACTION_NAMES[action]:8s} │ "
                      f"Weights: C={w[0]:.2f} E={w[1]:.2f} O={w[2]:.2f}")

            if info['solved']:
                if verbose:
                    print(f"\n  🎉 SOLVED in {step+1} moves!")
                    print(f"\n  Final state:")
                    print(env.render())
                    print(f"\n  Solution: {' → '.join(move_list)}")
                return True, step + 1, move_list

        if verbose:
            print(f"\n  ❌ Failed to solve in {max_moves} moves")
            print(f"\n  Final state:")
            print(env.render())

        return False, max_moves, move_list

    # ──────────────── Checkpointing ────────────────

    def _save(self, name: str):
        """Save all model parameters."""
        save_checkpoint(
            path=f"{self.config['save_dir']}/{name}.pt",
            encoder=self.encoder,
            agents=self.agents,
            composer=self.composer,
            optimizer=self.optimizer,
            curiosity_optimizer=self.curiosity_optimizer,
            config=self.config,
            stats={
                'total_steps': self.total_steps,
                'total_episodes': self.total_episodes,
                'best_solve_rate': self.best_solve_rate,
                'scramble_depth': self.current_scramble_depth,
            }
        )

    def load(self, path: str):
        """Load all model parameters from checkpoint."""
        stats = load_checkpoint(
            path=path,
            encoder=self.encoder,
            agents=self.agents,
            composer=self.composer,
            optimizer=self.optimizer,
            curiosity_optimizer=self.curiosity_optimizer,
            device=self.device,
        )
        if stats:
            self.total_steps = stats.get('total_steps', 0)
            self.total_episodes = stats.get('total_episodes', 0)
            self.best_solve_rate = stats.get('best_solve_rate', 0.0)
            self.current_scramble_depth = stats.get('scramble_depth',
                                                     self.config['initial_scramble_depth'])
        print(f"  Loaded checkpoint from {path}")
        print(f"    Steps: {self.total_steps:,}, Episodes: {self.total_episodes:,}")
        print(f"    Best solve rate: {self.best_solve_rate:.1%}")
        print(f"    Scramble depth: {self.current_scramble_depth}")


# Need F for mse_loss
import torch.nn.functional as F
