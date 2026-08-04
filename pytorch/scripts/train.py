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

    parser.add_argument("--learning-rate", type=float, default=0.00025,
                         help="Adam learning rate for the main model's optimizer.")
    parser.add_argument("--memory-size", type=int, default=10 ** 6,
                         help="Number of frames the replay memory ring buffer holds.")
    parser.add_argument("--memory-warmup", type=int, default=50_000,
                         help="Number of frames to collect before training/logging starts.")
    parser.add_argument("--batch-size", type=int, default=32,
                         help="Number of transitions sampled from replay memory per update step.")
    parser.add_argument("--max-train-frames", type=int, default=60_000,
                         help="Total number of frames to train for.")
    parser.add_argument("--update-main-freq", type=int, default=4,
                         help="Train the main model every N frames.")
    parser.add_argument("--update-target-freq", type=int, default=10_000,
                         help="Sync the target model's weights from the main model every N frames.")
    parser.add_argument("--log-freq", type=int, default=2_000,
                         help="Print a training progress line every N frames.")
    parser.add_argument("--average-loss-freq", type=int, default=400,
                         help="Average and record the loss over the last N frames.")
    parser.add_argument("--discount", type=float, default=0.99,
                         help="Discount factor (gamma) used in the Q-learning target.")

    parser.add_argument("--resume-from", type=str, default=None,
                         help="Path to a checkpoint directory to resume training from.")
    parser.add_argument("--save-path", type=str, default="pytorch/scripts/output",
                         help="Path to save the checkpoint to after training.")
    parser.add_argument("--metrics-dir", type=str, default="pytorch/scripts/output/metrics",
                         help="Directory to write episodes.csv/losses.csv to incrementally during training.")

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
        metrics_dir=args.metrics_dir,
    )

    print(f"Using device: {agent.device}")

    if args.resume_from:
        agent.load(args.resume_from)

    agent.train()
    agent.save(args.save_path)


if __name__ == "__main__":
    main()
