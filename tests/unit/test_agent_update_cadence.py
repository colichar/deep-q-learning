"""Optimizer updates follow collected transitions, including within vector ticks."""
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.agent.agent import SpaceInvaderAgent


def _training_agent(num_envs, ticks, update_freq, memory_warmup=0,
                    completed_transitions=0, replay_count=0):
    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.num_envs = num_envs
    agent.vec_env = None
    agent.Preprocessor = Mock()
    states = torch.zeros(num_envs, 4, 84, 84, dtype=torch.uint8)
    frames = torch.zeros(num_envs, 84, 84, dtype=torch.uint8)
    info = {"lives": np.full(num_envs, 3)}
    agent.Preprocessor.initialize_state_vec.return_value = (states, info)
    agent.Preprocessor.step_with_skip_vec.return_value = (
        None, np.zeros(num_envs), np.zeros(num_envs, dtype=bool),
        np.zeros(num_envs, dtype=bool), info,
    )
    agent.Preprocessor.new_state_vec.return_value = (states, frames)
    agent.ReplayMemory = Mock()
    agent.ReplayMemory.count = replay_count

    def add_frames(*args, **kwargs):
        agent.ReplayMemory.count += num_envs

    agent.ReplayMemory.add_frames.side_effect = add_frames
    agent.ExploreVsExploit = Mock(return_value=np.zeros(num_envs, dtype=int))
    agent.update_step = Mock(return_value=torch.tensor(0.5))
    agent.metrics_dir = agent.checkpoint_path = None
    agent.memory_warmup = memory_warmup
    agent.update_main_freq = update_freq
    agent.update_target_freq = agent.average_loss_freq = agent.log_freq = 10_000
    agent.completed_transitions = completed_transitions
    agent.max_train_transitions = ticks * num_envs
    agent.transition_nums = []
    agent.averaged_losses = []
    agent.losses = []
    agent.rewards = []
    agent._logged_reward_idx = 0
    agent.cumulative_wall_clock_seconds = 0.0
    agent._replay_restored_unmarked = False
    return agent


@pytest.mark.parametrize("num_envs, ticks, update_freq, expected_updates", [
    (1, 8, 4, 2),
    (8, 1, 4, 2),
    (3, 3, 4, 2),
    (5, 2, 3, 3),
])
def test_updates_once_per_crossed_threshold(num_envs, ticks, update_freq, expected_updates):
    agent = _training_agent(num_envs, ticks, update_freq)

    agent.train()

    assert agent.completed_transitions == num_envs * ticks
    assert agent.update_step.call_count == expected_updates
    assert len(agent.losses) == expected_updates


@pytest.mark.parametrize("warmup, expected_updates", [(0, 2), (4, 1), (5, 1), (8, 0)])
def test_thresholds_at_or_before_warmup_are_not_replayed(warmup, expected_updates):
    agent = _training_agent(8, 1, 4, memory_warmup=warmup)

    agent.train()

    assert agent.update_step.call_count == expected_updates


def test_empty_replay_on_resume_does_not_catch_up_missed_updates():
    agent = _training_agent(3, 3, 4, memory_warmup=4, completed_transitions=100)

    agent.train()

    assert agent.completed_transitions == 109
    assert agent.update_step.call_count == 1  # only transition 108, not 104 or earlier


def test_repeated_sessions_continue_update_thresholds_without_restart():
    agent = _training_agent(3, 1, 4)

    agent.train()
    agent.train()
    agent.train()

    assert agent.completed_transitions == 9
    assert agent.update_step.call_count == 2
