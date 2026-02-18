# MARVELS — Multi-Agent Residual Vision-Enhanced Learning for Symbolic Reasoning

> A novel reinforcement learning algorithm to solve the 3×3 Rubik's Cube **without any search algorithms** (no A*, MCTS, etc.) — pure policy learning with multi-agent coordination.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Architecture** | 3 specialized agents (Corner, Edge, Center) each with PPO actor-critic networks |
| **Skill Composer** | Attention-based dynamic policy blending — learns *when* to prioritize each agent |
| **Quaternion Encoding** | Geometrically meaningful 270-dim state vector using cubie rotation quaternions |
| **ICM Curiosity** | Intrinsic Curiosity Modules drive exploration in sparse-reward environments |
| **Curriculum Learning** | Starts with 1-move scrambles, increases difficulty as the agent improves |
| **Zero Search** | Pure policy inference at test time — no tree search, no backtracking |

## 🏗 Architecture

```
Cube State (54 stickers)
       ↓
Quaternion Encoder → 270-dim state vector
       ↓
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Corner   │ │ Edge     │ │ Center   │
│ Agent    │ │ Agent    │ │ Agent    │
│ (PPO+ICM)│ │ (PPO+ICM)│ │ (PPO+ICM)│
└────┬─────┘ └────┬─────┘ └────┬─────┘
     ↓            ↓            ↓
┌─────────────────────────────────────┐
│         Skill Composer              │
│  Attention-based policy blending    │
│  w_corner + w_edge + w_center = 1   │
└───────────────┬─────────────────────┘
                ↓
     Final Policy (18 actions)
```

## 📂 Project Structure

```
├── rubiks_env.py          # Complete 3×3 cube simulator (18 moves)
├── quaternion_encoder.py  # Quaternion state encoding (270-dim)
├── agents.py              # 3 Actor-Critic agents + ICM curiosity
├── skill_composer.py      # Attention-based policy blending
├── marvels_trainer.py     # Full PPO training loop with GAE
├── main.py                # Entry point: train + solve demo
├── utils.py               # Logging, saving, visualization
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Full training (default: 200 iterations)
python main.py

# Quick smoke test (5 iterations)
python main.py --episodes 5

# Train on GPU with custom settings
python main.py --mode train --episodes 1000 --device cuda --num-envs 32

# Resume from checkpoint
python main.py --mode train --checkpoint checkpoints/best_model.pt
```

### Solve Demo

```bash
# Solve a scrambled cube (loads best checkpoint)
python main.py --mode solve --scramble 15

# Custom checkpoint
python main.py --mode solve --checkpoint checkpoints/best_model.pt --scramble 10
```

### Evaluation

```bash
# Evaluate solve rate across difficulty levels
python main.py --mode eval --trials 100 --scramble 25
```

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `train` | `train`, `solve`, or `eval` |
| `--episodes` | `200` | Number of training iterations |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--num-envs` | `16` | Parallel environments for rollouts |
| `--lr` | `3e-4` | Learning rate |
| `--scramble` | `15` | Scramble depth for solve/eval |
| `--seed` | `42` | Random seed |

## 🧠 Algorithm Details

### Reward System

```
Total Reward = External + 0.5 × Intrinsic Curiosity

External:
  +100  if cube is solved
  -0.01 per move (efficiency pressure)

Intrinsic (per agent):
  Forward model prediction error (ICM)
```

### Training Pipeline

1. **Collect rollouts** from 16 parallel cube environments
2. **Encode states** via quaternion encoder (288 → 270 dims)
3. **Get agent policies** — each agent outputs 18-action probability distribution
4. **Compose policies** — attention mechanism blends agent outputs
5. **Compute rewards** — external + 0.5 × curiosity
6. **GAE advantages** — γ=0.999, λ=0.95
7. **PPO update** — clip=0.2, entropy=0.01, 4 epochs per batch
8. **Curriculum** — increase scramble depth when solve rate > 50%

### 18 Actions

6 faces × 3 rotations = 18 actions:
- **Faces**: U (Up), D (Down), F (Front), B (Back), L (Left), R (Right)
- **Rotations**: CW (90°), 180°, CCW (270°)

## 📊 Training Output

The training loop prints progress in this format:

```
  Iter    10/200 │ Scramble:  1 │ Solve: 45.0% │ Reward:  +23.45 │ Moves:  12.3 │ ...
  Iter    20/200 │ Scramble:  1 │ Solve: 78.0% │ Reward:  +67.89 │ Moves:   8.1 │ ...
  📈 Curriculum advanced → scramble depth = 2
  Iter    30/200 │ Scramble:  2 │ Solve: 32.0% │ Reward:  +15.67 │ Moves:  23.4 │ ...
```

## 📜 License

MIT
