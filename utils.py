"""
utils.py — Utility Functions for MARVELS
==========================================
Helpers for logging, checkpointing, visualization, and metrics tracking.
"""

import torch
import os
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque


# ──────────────── Moving Average Tracker ────────────────

class MovingAverage:
    """
    Exponential moving average tracker for metrics.
    Useful for smoothing noisy training signals.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.values = deque(maxlen=window)

    def add(self, value: float):
        self.values.append(value)

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        m = self.mean
        return (sum((v - m) ** 2 for v in self.values) / len(self.values)) ** 0.5

    @property
    def count(self) -> int:
        return len(self.values)

    def __repr__(self):
        return f"MovingAverage(mean={self.mean:.4f}, n={self.count})"


# ──────────────── Checkpointing ────────────────

def save_checkpoint(path: str, encoder, agents, composer,
                    optimizer, curiosity_optimizer,
                    config: Dict, stats: Dict):
    """
    Save all model parameters and training state to a file.

    Saves:
      - Encoder state dict
      - Each agent's state dict (actor-critic + curiosity)
      - Skill composer state dict
      - Optimizer states
      - Config and training stats
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    checkpoint = {
        'encoder': encoder.state_dict(),
        'agents': [agent.state_dict() for agent in agents],
        'composer': composer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'curiosity_optimizer': curiosity_optimizer.state_dict(),
        'config': config,
        'stats': stats,
        'timestamp': time.time(),
    }

    torch.save(checkpoint, path)


def load_checkpoint(path: str, encoder, agents, composer,
                    optimizer=None, curiosity_optimizer=None,
                    device=None) -> Optional[Dict]:
    """
    Load model parameters from a checkpoint file.

    Returns the stats dict from the checkpoint, or None if loading fails.
    """
    if not os.path.exists(path):
        print(f"  ⚠️  Checkpoint not found: {path}")
        return None

    map_location = device if device else 'cpu'
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    encoder.load_state_dict(checkpoint['encoder'])
    for i, agent in enumerate(agents):
        agent.load_state_dict(checkpoint['agents'][i])
    composer.load_state_dict(checkpoint['composer'])

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if curiosity_optimizer and 'curiosity_optimizer' in checkpoint:
        curiosity_optimizer.load_state_dict(checkpoint['curiosity_optimizer'])

    return checkpoint.get('stats', {})


# ──────────────── Time Formatting ────────────────

def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


# ──────────────── Cube Visualization ────────────────

# ANSI color codes for cube face colors
ANSI_COLORS = {
    0: '\033[97m■\033[0m',   # White  (Up)
    1: '\033[93m■\033[0m',   # Yellow (Down)
    2: '\033[91m■\033[0m',   # Red    (Front)
    3: '\033[38;5;208m■\033[0m',  # Orange (Back)
    4: '\033[92m■\033[0m',   # Green  (Left)
    5: '\033[94m■\033[0m',   # Blue   (Right)
}

PLAIN_COLORS = {
    0: 'W',   # White  (Up)
    1: 'Y',   # Yellow (Down)
    2: 'R',   # Red    (Front)
    3: 'O',   # Orange (Back)
    4: 'G',   # Green  (Left)
    5: 'B',   # Blue   (Right)
}


def visualize_cube_compact(faces: np.ndarray, use_color: bool = True) -> str:
    """
    Render cube faces as a compact cross-shaped net.

    Layout:
          U U U
          U U U
          U U U
    L L L F F F R R R B B B
    L L L F F F R R R B B B
    L L L F F F R R R B B B
          D D D
          D D D
          D D D

    Args:
        faces: (6, 3, 3) numpy array
        use_color: use ANSI color codes (True) or plain letters (False)
    """
    colors = ANSI_COLORS if use_color else PLAIN_COLORS

    # U=0, D=1, F=2, B=3, L=4, R=5
    def c(face_idx, r, c_idx):
        return colors[int(faces[face_idx, r, c_idx])]

    lines = []
    pad = '         ' if not use_color else '            '

    # Top face (U)
    for r in range(3):
        row = ' '.join(c(0, r, col) for col in range(3))
        lines.append(pad + row)

    # Middle band: L F R B
    for r in range(3):
        row = (
            ' '.join(c(4, r, col) for col in range(3)) + '  ' +
            ' '.join(c(2, r, col) for col in range(3)) + '  ' +
            ' '.join(c(5, r, col) for col in range(3)) + '  ' +
            ' '.join(c(3, r, col) for col in range(3))
        )
        lines.append(row)

    # Bottom face (D)
    for r in range(3):
        row = ' '.join(c(1, r, col) for col in range(3))
        lines.append(pad + row)

    return '\n'.join(lines)


# ──────────────── Training History Logger ────────────────

class TrainingLogger:
    """
    Logs training metrics to console and optionally to a JSON file.
    """

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.entries = []

    def log(self, iteration: int, metrics: Dict[str, Any]):
        """Add a log entry."""
        entry = {
            'iteration': iteration,
            'timestamp': time.time(),
            **metrics,
        }
        self.entries.append(entry)

        if self.log_file:
            self._write()

    def _write(self):
        """Write all entries to JSON file."""
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(self.entries, f, indent=2, default=str)

    def get_metric(self, key: str) -> List:
        """Get a list of values for a specific metric across all entries."""
        return [e.get(key) for e in self.entries if key in e]


# ──────────────── Seed Utilities ────────────────

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────── Device Detection ────────────────

def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Auto-detect best available device.
    Prefers CUDA GPU, falls back to MPS (Apple Silicon), then CPU.
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"  🖥  Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"  🖥  Using Apple Silicon MPS")
            return device

    device = torch.device('cpu')
    print(f"  🖥  Using CPU")
    return device


# ──────────────── Solve Rate Evaluator ────────────────

def evaluate_solve_rate(trainer, scramble_depth: int,
                        num_trials: int = 50,
                        max_moves: int = 200) -> Dict[str, float]:
    """
    Evaluate the current model's solve rate on fresh scrambles.

    Args:
        trainer: MARVELSTrainer instance
        scramble_depth: number of scramble moves
        num_trials: number of solve attempts
        max_moves: max moves per attempt

    Returns:
        dict with solve_rate, avg_moves (for solved), success_count
    """
    solved_count = 0
    total_moves = 0

    for trial in range(num_trials):
        solved, moves, _ = trainer.solve(
            scramble_depth=scramble_depth,
            max_moves=max_moves,
            verbose=False
        )
        if solved:
            solved_count += 1
            total_moves += moves

    solve_rate = solved_count / num_trials
    avg_moves = total_moves / max(solved_count, 1)

    return {
        'solve_rate': solve_rate,
        'avg_moves': avg_moves,
        'success_count': solved_count,
        'total_trials': num_trials,
        'scramble_depth': scramble_depth,
    }
