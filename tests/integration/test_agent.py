"""
Integration tests for SpaceInvaderAgent: real gym/ALE env, real model, real filesystem.
Slower than the ReplayMemory unit tests - run with `uv run pytest -m integration`
or skip with `uv run pytest -m "not integration"`.
"""
import numpy as np
import pytest

from src.agent.agent import SpaceInvaderAgent

pytestmark = pytest.mark.integration

AGENT_KWARGS = dict(
    memory_size=2000,
    memory_warmup=200,
    batch_size=16,
    max_train_frames=600,
    update_main_freq=4,
    update_target_freq=250,
    log_freq=200,
    average_loss_freq=100,
)


def test_training_runs_end_to_end():
    agent = SpaceInvaderAgent(**AGENT_KWARGS)
    agent.train()

    assert len(agent.averaged_losses) > 0
    assert len(agent.rewards) > 0


def test_training_writes_metrics_csvs_incrementally(tmp_path):
    metrics_dir = tmp_path / "metrics"
    agent = SpaceInvaderAgent(**AGENT_KWARGS, metrics_dir=str(metrics_dir))
    agent.train()

    episodes_csv = metrics_dir / "episodes.csv"
    losses_csv = metrics_dir / "losses.csv"
    assert episodes_csv.exists()
    assert losses_csv.exists()

    episode_rows = episodes_csv.read_text().splitlines()
    assert episode_rows[0] == "frame_num,episode_num,episode_reward,epsilon,wall_clock_elapsed_seconds"
    assert len(episode_rows) > 1

    loss_rows = losses_csv.read_text().splitlines()
    assert loss_rows[0] == "frame_num,avg_loss"
    assert len(loss_rows) > 1


def test_train_writes_periodic_checkpoint_resumable(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    agent = SpaceInvaderAgent(**AGENT_KWARGS, checkpoint_freq=250, checkpoint_path=str(checkpoint_path))
    agent.train()

    # proves checkpointing fired on its own mid-run, without an explicit agent.save() call
    history_dir = checkpoint_path / "history"
    mid_run_files = list(history_dir.glob("train_history_*"))
    assert len(mid_run_files) == 1
    mid_run_frame = int(mid_run_files[0].name.rsplit("_", 1)[-1])
    assert 0 < mid_run_frame < agent.max_train_frames

    # mirrors train.py's final save(), reproducing the multi-file case load_train_history handles
    agent.save(str(checkpoint_path))
    assert len(list(history_dir.glob("train_history_*"))) > 1

    resumed = SpaceInvaderAgent(**AGENT_KWARGS)
    resumed.load(str(checkpoint_path))

    assert resumed.start_frame_num == agent.frame_nums[-1]
    assert resumed.start_frame_num > mid_run_frame

    resumed.train()
    assert len(resumed.rewards) > 0


def test_agent_resumes_from_saved_checkpoint(tmp_path):
    agent = SpaceInvaderAgent(**AGENT_KWARGS)
    agent.train()

    losses_before = len(agent.averaged_losses)
    rewards_before = len(agent.rewards)
    last_frame_before = agent.frame_nums[-1]

    agent.save(str(tmp_path))

    resumed = SpaceInvaderAgent(**AGENT_KWARGS)
    resumed.load(str(tmp_path))

    assert np.array_equal(resumed.ReplayMemory.frames, agent.ReplayMemory.frames)
    assert resumed.ReplayMemory.idx == agent.ReplayMemory.idx
    assert resumed.ReplayMemory.count == agent.ReplayMemory.count
    assert resumed.start_frame_num == last_frame_before

    # Continuing training should run without error and append further history.
    resumed.train()

    assert len(resumed.averaged_losses) > losses_before
    assert len(resumed.rewards) > rewards_before
