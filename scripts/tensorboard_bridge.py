"""
Optional add-on: tails a training run's losses.csv/episodes.csv and mirrors them into
TensorBoard scalars, so you get live-updating graphs instead of reading log lines/CSVs.

Fully decoupled from train.py/agent.py — it only reads the CSVs that --metrics-dir already
writes, so it can be pointed at a run that's already in progress (or already finished) with
no changes to the training code, and dropping this file removes the feature entirely.

Usage:
    uv run python scripts/tensorboard_bridge.py --metrics-dir scripts/output-rmsprop/metrics
    uv run tensorboard --logdir scripts/output-rmsprop/tensorboard
"""
import argparse
import csv
import os
import time

from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Mirror a training run's CSV metrics into TensorBoard.")

    parser.add_argument("--metrics-dir", type=str, required=True,
                         help="The --metrics-dir a train.py run is writing losses.csv/episodes.csv to.")
    parser.add_argument("--logdir", type=str, default=None,
                         help="TensorBoard log directory. Defaults to a 'tensorboard' folder next to --metrics-dir.")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                         help="Seconds between checking the CSVs for new rows.")
    parser.add_argument("--reward-avg-window", type=int, default=20,
                         help="Number of trailing episodes averaged into each reward point.")
    parser.add_argument("--once", action="store_true",
                         help="Read whatever rows currently exist, write them, and exit (no polling loop).")

    return parser.parse_args()


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def sync_losses(path, writer, last_frame):
    new_last = last_frame
    for row in read_rows(path):
        frame_num = int(row["frame_num"])
        if frame_num <= last_frame:
            continue
        writer.add_scalar("loss/avg_loss", float(row["avg_loss"]), frame_num)
        new_last = frame_num
    return new_last


def sync_rewards(path, writer, last_frame, window):
    rows = read_rows(path)
    new_last = last_frame
    for i, row in enumerate(rows):
        frame_num = int(row["frame_num"])
        if frame_num <= last_frame:
            continue
        trailing = rows[max(0, i - window + 1):i + 1]
        avg_reward = sum(float(r["episode_reward"]) for r in trailing) / len(trailing)
        writer.add_scalar("reward/avg_episode_reward", avg_reward, frame_num)
        new_last = frame_num
    return new_last


def main():
    args = parse_args()

    logdir = args.logdir or os.path.join(os.path.dirname(os.path.normpath(args.metrics_dir)), "tensorboard")
    writer = SummaryWriter(log_dir=logdir)
    print(f"Writing TensorBoard scalars to '{logdir}'.")
    print(f"Watching '{args.metrics_dir}' every {args.poll_interval}s (Ctrl+C to stop). "
          f"Run `uv run tensorboard --logdir {logdir}` in another terminal to view.")

    losses_path = os.path.join(args.metrics_dir, "losses.csv")
    episodes_path = os.path.join(args.metrics_dir, "episodes.csv")
    last_loss_frame = -1
    last_reward_frame = -1

    while True:
        last_loss_frame = sync_losses(losses_path, writer, last_loss_frame)
        last_reward_frame = sync_rewards(episodes_path, writer, last_reward_frame, args.reward_avg_window)
        writer.flush()

        if args.once:
            break
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
