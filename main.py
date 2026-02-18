#!/usr/bin/env python3
"""
main.py — MARVELS Entry Point
===============================
Train and evaluate the MARVELS (Multi-Agent Residual Vision-Enhanced
Learning for Symbolic Reasoning) algorithm for solving the 3×3 Rubik's Cube.

Usage:
  python main.py                       # Train with defaults
  python main.py --mode train          # Full training
  python main.py --mode solve          # Solve demo (load best model)
  python main.py --mode eval           # Evaluate solve rate
  python main.py --episodes 5          # Quick smoke test

No search algorithms are used — pure policy learning + multi-agent coordination.
"""

import argparse
import sys
import os
import torch
import numpy as np

from rubiks_env import RubiksCube, NUM_ACTIONS, ACTION_NAMES
from quaternion_encoder import QuaternionEncoder
from agents import CornerAgent, EdgeAgent, CenterAgent
from skill_composer import SkillComposer
from marvels_trainer import MARVELSTrainer
from utils import get_device, set_seed, evaluate_solve_rate, format_time


def parse_args():
    parser = argparse.ArgumentParser(
        description='MARVELS — Multi-Agent RL for Rubik\'s Cube',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Train with defaults
  python main.py --mode train --episodes 500 --device cuda
  python main.py --mode solve --checkpoint checkpoints/best_model.pt
  python main.py --mode eval --scramble 10 --trials 100
        """
    )
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'solve', 'eval'],
                        help='Mode: train, solve, or eval (default: train)')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of training iterations (default: 200)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, mps (default: auto)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file for solve/eval modes')
    parser.add_argument('--scramble', type=int, default=15,
                        help='Scramble depth for solve/eval (default: 15)')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of evaluation trials (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num-envs', type=int, default=16,
                        help='Number of parallel environments (default: 16)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Print stats every N iterations (default: 10)')

    return parser.parse_args()


def print_banner():
    """Print the MARVELS banner."""
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                          ║")
    print("║     ███╗   ███╗ █████╗ ██████╗ ██╗   ██╗███████╗██╗     ║")
    print("║     ████╗ ████║██╔══██╗██╔══██╗██║   ██║██╔════╝██║     ║")
    print("║     ██╔████╔██║███████║██████╔╝██║   ██║█████╗  ██║     ║")
    print("║     ██║╚██╔╝██║██╔══██║██╔══██╗╚██╗ ██╔╝██╔══╝  ██║     ║")
    print("║     ██║ ╚═╝ ██║██║  ██║██║  ██║ ╚████╔╝ ███████╗███████╗║")
    print("║     ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝╚══════╝║")
    print("║                                                          ║")
    print("║   Multi-Agent Residual Vision-Enhanced Learning          ║")
    print("║   for Symbolic Reasoning                                 ║")
    print("║                                                          ║")
    print("║   🧊 Search-Free Rubik's Cube Solver                    ║")
    print("║   🤖 3 Specialized Agents + Skill Composer              ║")
    print("║   🧠 Quaternion State Encoding + ICM Curiosity           ║")
    print("║                                                          ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()


def print_architecture():
    """Print model architecture summary."""
    print("  Architecture Summary:")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  Cube State (54 stickers)                          │")
    print("  │       ↓                                            │")
    print("  │  Quaternion Encoder → 270-dim state vector          │")
    print("  │       ↓                                            │")
    print("  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │")
    print("  │  │ Corner   │ │ Edge     │ │ Center   │           │")
    print("  │  │ Agent    │ │ Agent    │ │ Agent    │           │")
    print("  │  │ (PPO+ICM)│ │ (PPO+ICM)│ │ (PPO+ICM)│           │")
    print("  │  └────┬─────┘ └────┬─────┘ └────┬─────┘           │")
    print("  │       ↓            ↓            ↓                  │")
    print("  │  ┌─────────────────────────────────────┐           │")
    print("  │  │         Skill Composer              │           │")
    print("  │  │  Attention-based policy blending     │           │")
    print("  │  │  w_corner + w_edge + w_center = 1    │           │")
    print("  │  └───────────────┬─────────────────────┘           │")
    print("  │                  ↓                                 │")
    print("  │       Final Policy (18 actions)                    │")
    print("  └─────────────────────────────────────────────────────┘")
    print()


def mode_train(args):
    """Run training mode."""
    print("  Mode: TRAINING")
    print()
    print_architecture()

    # Device setup
    if args.device == 'auto':
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)
        print(f"  🖥  Using {args.device}")

    # Set seed
    set_seed(args.seed)
    print(f"  🌱 Seed: {args.seed}")

    # Configuration
    config = {
        'num_envs': args.num_envs,
        'lr': args.lr,
        'device': str(device),
        'save_dir': args.save_dir,
        'rollout_length': 128,
        'ppo_epochs': 4,
        'mini_batch_size': 64,
        'clip_ratio': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'curiosity_coef': 0.5,
        'max_grad_norm': 0.5,
        'gamma': 0.999,
        'gae_lambda': 0.95,
        'curiosity_lr': 1e-3,
        'initial_scramble_depth': 1,
        'max_scramble_depth': 25,
        'curriculum_threshold': 0.5,
        'max_moves': 200,
    }

    # Load from checkpoint if provided
    trainer = MARVELSTrainer(config)

    if args.checkpoint:
        trainer.load(args.checkpoint)

    # Count parameters
    total_params = sum(
        p.numel() for p in trainer.encoder.parameters()
    )
    for agent in trainer.agents:
        total_params += sum(p.numel() for p in agent.parameters())
    total_params += sum(p.numel() for p in trainer.composer.parameters())
    print(f"  📊 Total parameters: {total_params:,}")
    print()

    # Train
    history = trainer.train(
        num_iterations=args.episodes,
        log_interval=args.log_interval,
        save_interval=max(args.episodes // 10, 1),
    )

    # Final evaluation
    print("\n  Running final evaluation...")
    for depth in [1, 3, 5, 8, 10, 15]:
        results = evaluate_solve_rate(trainer, depth, num_trials=20)
        status = "✅" if results['solve_rate'] > 0 else "❌"
        print(f"    {status} Scramble {depth:2d}: "
              f"{results['solve_rate']:.0%} solved, "
              f"avg {results['avg_moves']:.1f} moves")

    # Demo solve
    print("\n" + "=" * 60)
    print("  DEMO SOLVE")
    print("=" * 60)
    # Try solving with increasing difficulty until we fail
    for depth in [1, 3, 5, 8, 10, 15, 20]:
        solved, moves, move_list = trainer.solve(
            scramble_depth=depth, verbose=False
        )
        if solved:
            print(f"  ✅ Scramble {depth:2d} → SOLVED in {moves} moves!")
        else:
            print(f"  ❌ Scramble {depth:2d} → Failed")
            break

    print("\n  Training complete! Checkpoints saved to:", args.save_dir)


def mode_solve(args):
    """Run solve demo mode."""
    print("  Mode: SOLVE DEMO")
    print()

    # Device
    if args.device == 'auto':
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)

    config = {'device': str(device), 'save_dir': args.save_dir}
    trainer = MARVELSTrainer(config)

    # Load checkpoint
    checkpoint_path = args.checkpoint or os.path.join(args.save_dir, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        print(f"  ❌ No checkpoint found at: {checkpoint_path}")
        print(f"     Run training first: python main.py --mode train")
        sys.exit(1)

    trainer.load(checkpoint_path)

    # Solve demo
    print("\n" + "=" * 60)
    print(f"  Solving cube scrambled with {args.scramble} moves")
    print("=" * 60)

    solved, moves, move_list = trainer.solve(
        scramble_depth=args.scramble,
        max_moves=200,
        verbose=True,
    )

    if solved:
        print(f"\n  🎉 SOLVED in {moves} moves!")
    else:
        print(f"\n  ❌ Could not solve in 200 moves.")
        print(f"     Try with a smaller scramble depth: --scramble 5")


def mode_eval(args):
    """Run evaluation mode."""
    print("  Mode: EVALUATION")
    print()

    # Device
    if args.device == 'auto':
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)

    config = {'device': str(device), 'save_dir': args.save_dir}
    trainer = MARVELSTrainer(config)

    # Load checkpoint
    checkpoint_path = args.checkpoint or os.path.join(args.save_dir, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        print(f"  ❌ No checkpoint found at: {checkpoint_path}")
        print(f"     Run training first: python main.py --mode train")
        sys.exit(1)

    trainer.load(checkpoint_path)

    # Evaluate across scramble depths
    print("\n" + "=" * 60)
    print(f"  Evaluating with {args.trials} trials per depth")
    print("=" * 60)
    print()
    print(f"  {'Depth':>5} │ {'Solve Rate':>10} │ {'Avg Moves':>10} │ {'Solved':>6}")
    print(f"  {'─'*5}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*6}")

    depths = [1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 25]
    for depth in depths:
        if depth > args.scramble:
            break
        results = evaluate_solve_rate(
            trainer, depth, num_trials=args.trials, max_moves=200
        )
        print(f"  {depth:5d} │ {results['solve_rate']:10.1%} │ "
              f"{results['avg_moves']:10.1f} │ "
              f"{results['success_count']:3d}/{results['total_trials']}")

    print()
    print("  Evaluation complete.")


def main():
    print_banner()
    args = parse_args()

    if args.mode == 'train':
        mode_train(args)
    elif args.mode == 'solve':
        mode_solve(args)
    elif args.mode == 'eval':
        mode_eval(args)


if __name__ == '__main__':
    main()
