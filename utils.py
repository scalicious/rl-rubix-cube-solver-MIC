"""
utils.py - Checkpointing, logging, visualization, and misc helpers.
"""

import torch
import os
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque


class MovingAverage:
    """Simple windowed average for tracking metrics."""

    def __init__(self, window=100):
        self.window = window
        self.values = deque(maxlen=window)

    def add(self, value):
        self.values.append(value)

    @property
    def mean(self):
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def std(self):
        if len(self.values) < 2:
            return 0.0
        m = self.mean
        return (sum((v - m) ** 2 for v in self.values) / len(self.values)) ** 0.5

    @property
    def count(self):
        return len(self.values)

    def __repr__(self):
        return f"MovingAverage(mean={self.mean:.4f}, n={self.count})"


def save_checkpoint(path, encoder, agents, composer,
                    optimizer, curiosity_optimizer, config, stats):
    """Save all model weights + optimizer state + training stats."""
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


def load_checkpoint(path, encoder, agents, composer,
                    optimizer=None, curiosity_optimizer=None, device=None):
    """Load model from checkpoint. Returns stats dict or None."""
    if not os.path.exists(path):
        print(f"  Warning: checkpoint not found: {path}")
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


def format_time(seconds):
    """Human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


# ANSI colors for terminal cube rendering
ANSI_COLORS = {
    0: '\033[97m■\033[0m',   # White
    1: '\033[93m■\033[0m',   # Yellow
    2: '\033[91m■\033[0m',   # Red
    3: '\033[38;5;208m■\033[0m',  # Orange
    4: '\033[92m■\033[0m',   # Green
    5: '\033[94m■\033[0m',   # Blue
}

PLAIN_COLORS = {
    0: 'W', 1: 'Y', 2: 'R', 3: 'O', 4: 'G', 5: 'B',
}


def visualize_cube_compact(faces, use_color=True):
    """Render cube as a cross-shaped net (text)."""
    colors = ANSI_COLORS if use_color else PLAIN_COLORS

    def c(face_idx, r, c_idx):
        return colors[int(faces[face_idx, r, c_idx])]

    lines = []
    pad = '         ' if not use_color else '            '

    # U face
    for r in range(3):
        row = ' '.join(c(0, r, col) for col in range(3))
        lines.append(pad + row)

    # L F R B band
    for r in range(3):
        row = (
            ' '.join(c(4, r, col) for col in range(3)) + '  ' +
            ' '.join(c(2, r, col) for col in range(3)) + '  ' +
            ' '.join(c(5, r, col) for col in range(3)) + '  ' +
            ' '.join(c(3, r, col) for col in range(3))
        )
        lines.append(row)

    # D face
    for r in range(3):
        row = ' '.join(c(1, r, col) for col in range(3))
        lines.append(pad + row)

    return '\n'.join(lines)


class TrainingLogger:
    """Logs training metrics, optionally to a JSON file."""

    def __init__(self, log_file=None):
        self.log_file = log_file
        self.entries = []

    def log(self, iteration, metrics):
        entry = {
            'iteration': iteration,
            'timestamp': time.time(),
            **metrics,
        }
        self.entries.append(entry)

        if self.log_file:
            self._write()

    def _write(self):
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(self.entries, f, indent=2, default=str)

    def get_metric(self, key):
        return [e.get(key) for e in self.entries if key in e]


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu=True):
    """Auto-detect best device (CUDA > MPS > CPU)."""
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"  Using CUDA: {torch.cuda.get_device_name(0)}")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  Using Apple Silicon (MPS)")
            return torch.device('mps')

    print(f"  Using CPU")
    return torch.device('cpu')


def evaluate_solve_rate(trainer, scramble_depth, num_trials=50, max_moves=200):
    """
    Run num_trials solve attempts at given scramble depth.
    Returns dict with solve_rate, avg_moves, etc.
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
