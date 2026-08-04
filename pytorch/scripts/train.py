"""
CLI entry point for kicking off a SpaceInvaderAgent training run.

Usage:
    uv run python pytorch/scripts/train.py --max-train-frames 70000
    uv run python pytorch/scripts/train.py --resume-from pytorch/scripts/output --save-path pytorch/scripts/output
"""
import argparse
import os
import sys

# Make `pytorch/` (the parent of this script's directory) importable as the
# root for `src...` regardless of the caller's cwd, since the repo also has a
# deprecated top-level `src/` (TensorFlow) that would otherwise shadow it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.agent import SpaceInvaderAgent  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SpaceInvaderAgent on ALE/SpaceInvaders-v5.")

    parser.add_argument("--learning-rate", type=float, default=0.00025)
    parser.add_argument("--memory-size", type=int, default=10 ** 6)
    parser.add_argument("--memory-warmup", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-frames", type=float, default=0.6 * 10 ** 4)
    parser.add_argument("--update-main-freq", type=int, default=4)
    parser.add_argument("--update-target-freq", type=int, default=10_000)
    parser.add_argument("--log-freq", type=float, default=0.2 * 10 ** 4)
    parser.add_argument("--average-loss-freq", type=int, default=400)
    parser.add_argument("--discount", type=float, default=0.99)

    parser.add_argument("--resume-from", type=str, default=None,
                         help="Path to a checkpoint directory to resume training from.")
    parser.add_argument("--save-path", type=str, default="pytorch/scripts/output",
                         help="Path to save the checkpoint to after training.")

    return parser.parse_args()


def main():
    args = parse_args()

    agent = SpaceInvaderAgent(
        learning_rate=args.learning_rate,
        memory_size=args.memory_size,
        memory_warmup=args.memory_warmup,
        batch_size=args.batch_size,
        max_train_frames=args.max_train_frames,
        update_main_freq=args.update_main_freq,
        update_target_freq=args.update_target_freq,
        log_freq=args.log_freq,
        average_loss_freq=args.average_loss_freq,
        discount=args.discount,
    )

    print(f"Using device: {agent.device}")

    if args.resume_from:
        agent.load(args.resume_from)

    agent.train()
    agent.save(args.save_path)


if __name__ == "__main__":
    main()
