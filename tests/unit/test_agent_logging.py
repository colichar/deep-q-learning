"""Loss averaging and progress logging use independent transition schedules."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.agent.agent import SpaceInvaderAgent


class CountingReplay:
    def __init__(self):
        self.count = 0

    def add_frames(self, frames, actions, rewards, terminals, boundaries):
        self.count += len(actions)


def _agent(num_envs, max_train_transitions, average_loss_freq, log_freq, update_main_freq=1):
    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.num_envs = num_envs
    agent.vec_env = None
    agent.Preprocessor = Mock()
    states = torch.zeros(num_envs, 4, 84, 84, dtype=torch.uint8)
    frames = torch.zeros(num_envs, 84, 84, dtype=torch.uint8)
    info = {"lives": np.full(num_envs, 3)}
    agent.Preprocessor.initialize_state_vec.return_value = (states, info)
    agent.Preprocessor.step_with_skip_vec.return_value = (
        None, np.zeros(num_envs), np.zeros(num_envs, dtype=bool), np.zeros(num_envs, dtype=bool), info,
    )
    agent.Preprocessor.new_state_vec.return_value = (states, frames)
    agent.ReplayMemory = CountingReplay()
    agent.ExploreVsExploit = Mock(return_value=np.zeros(num_envs, dtype=int))
    agent.ExploreVsExploit.get_epsilon.return_value = 0.5
    agent.update_step = Mock(return_value=torch.tensor(0.5))
    agent.TargetModel = Mock()
    agent.memory_warmup = 0
    agent.update_main_freq = update_main_freq
    agent.update_target_freq = max_train_transitions + 100
    agent.average_loss_freq = average_loss_freq
    agent.log_freq = log_freq
    agent.max_train_transitions = max_train_transitions
    agent.completed_transitions = 0
    agent.checkpoint_path = None
    agent.metrics_dir = None
    agent.rewards = []
    agent.losses = []
    agent.transition_nums = []
    agent.averaged_losses = []
    agent._logged_reward_idx = 0
    agent._logged_loss_idx = 0
    agent.cumulative_wall_clock_seconds = 0.0
    agent._replay_restored_unmarked = False
    return agent


def test_logging_does_not_wait_for_loss_schedule_intersection(tmp_path, capsys):
    agent = _agent(1, 2_000, average_loss_freq=400, log_freq=1_000)
    agent.metrics_dir = str(tmp_path)

    agent.train()

    assert agent.transition_nums == [400, 800, 1_200, 1_600, 2_000]
    assert agent.averaged_losses == pytest.approx([0.5] * 5)
    assert (tmp_path / "losses.csv").read_text().splitlines() == [
        "transition_num,avg_loss", "1000,0.5", "2000,0.5",
    ]
    output = capsys.readouterr().out
    assert "Finished 1000 transitions. Loss: 0.500000" in output
    assert "Finished 2000 transitions. Loss: 0.500000" in output


def test_vector_threshold_crossings_and_no_new_average(tmp_path, capsys):
    agent = _agent(3, 15, average_loss_freq=4, log_freq=5)
    agent.metrics_dir = str(tmp_path)

    agent.train()

    assert agent.completed_transitions == 15
    assert agent.transition_nums == [6, 9, 12]
    assert (tmp_path / "losses.csv").read_text().splitlines() == [
        "transition_num,avg_loss", "6,0.5", "12,0.5",
    ]
    output = capsys.readouterr().out
    assert "Finished 6 transitions. Loss: 0.500000" in output
    assert "Finished 12 transitions. Loss: 0.500000" in output
    assert "Finished 15 transitions. Loss: n/a (no new loss average)" in output


def test_log_without_optimizer_updates_has_no_nan_or_loss_row(tmp_path, capsys):
    agent = _agent(1, 6, average_loss_freq=2, log_freq=3, update_main_freq=100)
    agent.metrics_dir = str(tmp_path)

    agent.train()

    assert agent.transition_nums == agent.averaged_losses == agent.losses == []
    assert (tmp_path / "losses.csv").read_text().splitlines() == ["transition_num,avg_loss"]
    output = capsys.readouterr().out
    assert output.count("Loss: n/a (no new loss average)") == 2
    assert "nan" not in output.lower()


def test_pending_average_survives_history_save_and_resume(tmp_path, capsys):
    agent = _agent(1, 4, average_loss_freq=4, log_freq=6)
    agent.train()
    agent.save_train_history(str(tmp_path / "history"))

    resumed = _agent(1, 2, average_loss_freq=4, log_freq=6)
    resumed.load_train_history(str(tmp_path / "history"))
    resumed.train()

    assert "Finished 6 transitions. Loss: 0.500000" in capsys.readouterr().out
