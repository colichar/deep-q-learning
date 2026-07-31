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
