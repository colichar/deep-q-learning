"""
CLI entry point for watching a trained SpaceInvaderAgent play, saving each episode as a gif.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --checkpoint scripts/output-rmsprop --episodes 3
"""
import argparse
import os
import sys

# Make the repo root (the parent of this script's directory) importable as the
# root for `src...` regardless of the caller's cwd.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.agent.agent import SpaceInvaderAgent  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained SpaceInvaderAgent checkpoint and record its gameplay as gifs."
    )

    parser.add_argument("--checkpoint", type=str, default="scripts/output-rmsprop-centered-50M",
                         help="Path to the checkpoint directory to load (must contain a 'model/' subfolder).")
    parser.add_argument("--optimizer", type=str, default="rmsprop", choices=["adam", "rmsprop"],
                         help="Optimizer the checkpoint was saved with; must match, since optimizer state "
                              "(e.g. RMSprop's centered squared-gradient average) isn't portable between "
                              "optimizer types. output-rmsprop-centered-* checkpoints were saved with 'rmsprop'.")
    parser.add_argument("--episodes", type=int, default=5,
                         help="Number of episodes to play and save as gifs.")

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)

    # Eval never touches the replay memory, so keep its capacity tiny rather than
    # eagerly allocating the multi-GB default (10**6 frames) for nothing.
    agent = SpaceInvaderAgent(optimizer=args.optimizer, memory_size=1_000)

    print(f"Using device: {agent.device}")
    print(f"Loading checkpoint from {checkpoint_path}...")
    agent.load(checkpoint_path)

    # export_as_gif writes to the cwd using a relative filename, so run from the repo
    # root regardless of where this script was invoked from, so gifs land there.
    os.chdir(REPO_ROOT)

    print(f"Playing {args.episodes} episode(s)...")
    rewards = agent.evaluate(args.episodes)
    agent.close()

    for i, reward in enumerate(rewards, start=1):
        print(f"Episode {i}: reward {reward} -> eval_{i}_{reward}.gif")


if __name__ == "__main__":
    main()
