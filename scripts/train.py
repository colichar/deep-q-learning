"""
CLI entry point for kicking off a SpaceInvaderAgent training run.

Usage:
    uv run python scripts/train.py --max-train-frames 70000
    uv run python scripts/train.py --resume-from scripts/output --save-path scripts/output
"""
import argparse
import os
import sys

# Make the repo root (the parent of this script's directory) importable as the
# root for `src...` regardless of the caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.agent import SpaceInvaderAgent  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SpaceInvaderAgent on ALE/SpaceInvaders-v5.")

    parser.add_argument("--learning-rate", type=float, default=0.00025,
                         help="Learning rate for the main model's optimizer.")
    parser.add_argument("--optimizer", type=str, default="rmsprop", choices=["adam", "rmsprop"],
                         help="Optimizer for the main model. 'rmsprop' uses the DeepMind Nature paper's "
                              "centered RMSProp (gradient momentum / squared gradient momentum 0.95, min "
                              "squared gradient 0.01), not PyTorch's RMSprop defaults. Note: "
                              "--resume-from requires the same optimizer as the checkpoint being loaded, "
                              "since optimizer state (e.g. Adam's moment estimates) isn't portable "
                              "between optimizer types.")
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
    parser.add_argument("--log-freq", type=int, default=10_000,
                         help="Print a training progress line every N frames.")
    parser.add_argument("--average-loss-freq", type=int, default=400,
                         help="Average and record the loss over the last N frames.")
    parser.add_argument("--discount", type=float, default=0.99,
                         help="Discount factor (gamma) used in the Q-learning target.")
    parser.add_argument("--num-envs", type=int, default=1,
                         help="Number of parallel ALE sub-envs to train against. 1 uses a "
                              "SyncVectorEnv (single-env behavior); >1 uses a subprocess-based "
                              "AsyncVectorEnv. --memory-size is split evenly across sub-envs, "
                              "not multiplied by N.")

    parser.add_argument("--resume-from", type=str, default=None,
                         help="Path to a checkpoint directory to resume training from.")
    parser.add_argument("--save-path", type=str, default="scripts/output",
                         help="Path to save the checkpoint to after training.")
    parser.add_argument("--metrics-dir", type=str, default="scripts/output/metrics",
                         help="Directory to write episodes.csv/losses.csv to incrementally during training.")
    parser.add_argument("--checkpoint-freq", type=int, default=25_000,
                         help="Save a checkpoint to --save-path every N frames during training.")
    parser.add_argument("--replay-checkpoint-freq", type=int, default=None,
                         help="Save the replay memory buffer (the expensive part of a checkpoint) "
                              "every N frames, independently of --checkpoint-freq. Defaults to "
                              "--checkpoint-freq. Must be a multiple of --checkpoint-freq, since "
                              "it's only checked when a checkpoint fires. Set higher than "
                              "--checkpoint-freq to checkpoint model/history often while writing "
                              "the multi-GB replay buffer less often; a resumed run will then load "
                              "a replay memory snapshot slightly behind the resumed frame count, "
                              "which is harmless.")

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
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_path=args.save_path,
        replay_checkpoint_freq=args.replay_checkpoint_freq,
        optimizer=args.optimizer,
        num_envs=args.num_envs,
    )

    print(f"Using device: {agent.device}")

    if args.resume_from:
        agent.load(args.resume_from)

    agent.train()
    agent.save(args.save_path)
    agent.close()


if __name__ == "__main__":
    main()
