#!/usr/bin/env python3
"""
main.py - Entry point for MARVELS training and evaluation.
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
        description='MARVELS - Multi-Agent RL for Rubik\'s Cube',
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


def mode_train(args):
    """Run training mode."""
    print("  Mode: TRAINING")
    print()

    # Device
    if args.device == 'auto':
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)
        print(f"  Using device: {args.device}")

    set_seed(args.seed)
    print(f"  Seed: {args.seed}")

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

    trainer = MARVELSTrainer(config)

    if args.checkpoint:
        trainer.load(args.checkpoint)

    # Count params
    total_params = sum(p.numel() for p in trainer.encoder.parameters())
    for agent in trainer.agents:
        total_params += sum(p.numel() for p in agent.parameters())
    total_params += sum(p.numel() for p in trainer.composer.parameters())
    print(f"  Total parameters: {total_params:,}")
    print()

    history = trainer.train(
        num_iterations=args.episodes,
        log_interval=args.log_interval,
        save_interval=max(args.episodes // 10, 1),
    )

    # Final eval across scramble depths
    print("\n  Running final evaluation...")
    for depth in [1, 3, 5, 8, 10, 15]:
        results = evaluate_solve_rate(trainer, depth, num_trials=20)
        status = "OK" if results['solve_rate'] > 0 else "FAIL"
        print(f"    [{status}] Scramble {depth:2d}: "
              f"{results['solve_rate']:.0%} solved, "
              f"avg {results['avg_moves']:.1f} moves")

    # Demo solve
    print("\n" + "=" * 60)
    print("  DEMO SOLVE")
    print("=" * 60)
    for depth in [1, 3, 5, 8, 10, 15, 20]:
        solved, moves, move_list = trainer.solve(
            scramble_depth=depth, verbose=False
        )
        if solved:
            print(f"  Scramble {depth:2d} -> SOLVED in {moves} moves")
        else:
            print(f"  Scramble {depth:2d} -> Failed")
            break

    print("\n  Training complete! Checkpoints saved to:", args.save_dir)


def mode_solve(args):
    """Solve demo — loads a trained model and tries to solve a scrambled cube."""
    print("  Mode: SOLVE DEMO")
    print()

    if args.device == 'auto':
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)

    config = {'device': str(device), 'save_dir': args.save_dir}
    trainer = MARVELSTrainer(config)

    checkpoint_path = args.checkpoint or os.path.join(args.save_dir, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        print(f"  No checkpoint found at: {checkpoint_path}")
        print(f"  Run training first: python main.py --mode train")
        sys.exit(1)

    trainer.load(checkpoint_path)

    print("\n" + "=" * 60)
    print(f"  Solving cube scrambled with {args.scramble} moves")
    print("=" * 60)

    solved, moves, move_list = trainer.solve(
        scramble_depth=args.scramble,
        max_moves=200,
        verbose=True,
    )

    if solved:
        print(f"\n  SOLVED in {moves} moves!")
    else:
        print(f"\n  Could not solve in 200 moves.")
        print(f"  Try with a smaller scramble depth: --scramble 5")


def mode_eval(args):
    """Evaluate solve rate across different scramble depths."""
    print("  Mode: EVALUATION")
    print()

    if args.device == 'auto':
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)

    config = {'device': str(device), 'save_dir': args.save_dir}
    trainer = MARVELSTrainer(config)

    checkpoint_path = args.checkpoint or os.path.join(args.save_dir, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        print(f"  No checkpoint found at: {checkpoint_path}")
        print(f"  Run training first: python main.py --mode train")
        sys.exit(1)

    trainer.load(checkpoint_path)

    print("\n" + "=" * 60)
    print(f"  Evaluating with {args.trials} trials per depth")
    print("=" * 60)
    print()
    print(f"  {'Depth':>5} | {'Solve Rate':>10} | {'Avg Moves':>10} | {'Solved':>6}")
    print(f"  {'─'*5}─+─{'─'*10}─+─{'─'*10}─+─{'─'*6}")

    depths = [1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 25]
    for depth in depths:
        if depth > args.scramble:
            break
        results = evaluate_solve_rate(
            trainer, depth, num_trials=args.trials, max_moves=200
        )
        print(f"  {depth:5d} | {results['solve_rate']:10.1%} | "
              f"{results['avg_moves']:10.1f} | "
              f"{results['success_count']:3d}/{results['total_trials']}")

    print()
    print("  Evaluation complete.")


def main():
    print()
    print("  MARVELS - Multi-Agent Rubik's Cube Solver")
    print()

    args = parse_args()

    if args.mode == 'train':
        mode_train(args)
    elif args.mode == 'solve':
        mode_solve(args)
    elif args.mode == 'eval':
        mode_eval(args)


if __name__ == '__main__':
    main()
