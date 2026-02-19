# MARVELS — Rubik's Cube Solver (No Search)

RL-based approach to solving the 3x3 Rubik's Cube without any search algorithms (no MCTS, no A*, nothing). Just pure policy learning with multi-agent coordination.

MARVELS = Multi-Agent Residual Vision-Enhanced Learning for Symbolic Reasoning

## What this does

Instead of using a single giant network, we split the problem across three specialized agents:
- **Corner Agent** — focuses on getting the 8 corners right
- **Edge Agent** — handles the 12 edge pieces
- **Center Agent** — deals with overall face orientation

Each agent has its own PPO actor-critic network plus an ICM curiosity module for exploration. A **Skill Composer** (attention-based) learns to blend their policies dynamically — kind of like how a human might focus on corners first, then edges, etc., except this is learned from scratch.

The state representation uses quaternion encodings of cubie rotations instead of flat one-hot stickers, which gives the network a more geometric view of the cube (270-dim vector).

## Project structure

```
rubiks_env.py          — Cube simulator, 18 moves (6 faces x 3 rotations)
quaternion_encoder.py  — Quaternion-based state encoding
agents.py              — The 3 actor-critic agents + curiosity modules
skill_composer.py      — Attention-based policy blending
marvels_trainer.py     — PPO training loop, GAE, curriculum learning
main.py                — CLI entry point
utils.py               — Checkpointing, logging, misc helpers
```

## Setup

Python 3.10+, PyTorch 2.0+

```bash
pip install -r requirements.txt
```

## Usage

```bash
# train (default 200 iterations, starts with easy 1-move scrambles)
python main.py

# quick test run
python main.py --episodes 5

# train on GPU
python main.py --episodes 1000 --device cuda --num-envs 32

# resume from checkpoint
python main.py --checkpoint checkpoints/best_model.pt

# try solving a scrambled cube
python main.py --mode solve --scramble 15

# evaluate solve rate across difficulties
python main.py --mode eval --trials 100 --scramble 25
```

## How training works

1. Collect rollouts from 16 parallel envs
2. Encode states with quaternion encoder (288 raw dims -> 270 projected)
3. Each agent produces an 18-action policy
4. Skill composer blends them with learned attention weights
5. Rewards = external (+100 solve, -0.01/move) + 0.5 * curiosity
6. GAE advantages (gamma=0.999, lambda=0.95)
7. PPO update with clipping (eps=0.2), 4 epochs per batch
8. Curriculum: bump scramble depth when solve rate > 50%

## Config

| Arg | Default | What it does |
|-----|---------|-------------|
| `--mode` | `train` | train / solve / eval |
| `--episodes` | `200` | training iterations |
| `--device` | `auto` | cpu / cuda / mps |
| `--num-envs` | `16` | parallel envs |
| `--lr` | `3e-4` | learning rate |
| `--scramble` | `15` | scramble depth (solve/eval) |
| `--seed` | `42` | random seed |

## License

MIT
