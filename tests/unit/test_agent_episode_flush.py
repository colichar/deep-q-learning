"""Only completed games contribute to episode statistics (issue #48)."""
import csv

import numpy as np
import pytest
import torch

from src.agent.agent import SpaceInvaderAgent


class FakePreprocessor:
    """Feeds prescripted rewards, lives, and truncations per env."""

    def __init__(self, num_envs, ticks):
        self.num_envs = num_envs
        self.ticks = iter(ticks)

    def initialize_state_vec(self, vec_env):
        state = torch.zeros(self.num_envs, 4, 84, 84, dtype=torch.uint8)
        return state, {"lives": np.full(self.num_envs, 3)}

    def step_with_skip_vec(self, vec_env, actions):
        rewards, lives, truncated = next(self.ticks)
        obs = np.zeros((self.num_envs, 210, 160, 3), dtype=np.uint8)
        return (obs, np.array(rewards, dtype=float), np.array(lives) == 0,
                np.array(truncated, dtype=bool), {"lives": np.array(lives)})

    def new_state_vec(self, new_raw_obs, curr_states):
        return curr_states, torch.zeros(self.num_envs, 84, 84, dtype=torch.uint8)


class FakeReplayMemory:
    count = 0  # always below memory_warmup, so update/target/loss-averaging never fire

    def add_frames(self, *args, **kwargs):
        pass


class FakeExploreVsExploit:
    def __init__(self, num_envs):
        self.num_envs = num_envs

    def __call__(self, curr_states, frame_num):
        return np.zeros(self.num_envs, dtype=int)

    def get_epsilon(self, frame_num):
        return 0.5


def _make_agent(num_envs, ticks, max_train_frames, metrics_dir=None):
    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.num_envs = num_envs
    agent.Preprocessor = FakePreprocessor(num_envs, ticks)
    agent.vec_env = None
    agent.ReplayMemory = FakeReplayMemory()
    agent.ExploreVsExploit = FakeExploreVsExploit(num_envs)
    agent.metrics_dir = str(metrics_dir) if metrics_dir else None
    agent.checkpoint_path = None
    agent.memory_warmup = 10 ** 9
    agent.update_main_freq = 10 ** 9
    agent.update_target_freq = 10 ** 9
    agent.average_loss_freq = 10 ** 9
    agent.log_freq = 10 ** 9
    agent.start_frame_num = 0
    agent.max_train_frames = max_train_frames
    agent.rewards = []
    agent.losses = []
    agent.frame_nums = []
    agent.averaged_losses = []
    agent._logged_reward_idx = 0
    agent.cumulative_wall_clock_seconds = 0.0
    agent._replay_restored_unmarked = False
    return agent


def _episode_rows(metrics_dir):
    with (metrics_dir / "episodes.csv").open(newline="") as file:
        return list(csv.DictReader(file))


def test_partial_episode_is_not_recorded_when_budget_runs_out(tmp_path):
    ticks = [([1.0], [3], [False])] * 3
    agent = _make_agent(num_envs=1, ticks=ticks, max_train_frames=3, metrics_dir=tmp_path)

    agent.train()

    assert agent.rewards == []
    assert _episode_rows(tmp_path) == []


@pytest.mark.parametrize("final_lives, final_truncated", [(0, False), (3, True)])
def test_completion_on_final_tick_is_recorded_once(tmp_path, final_lives, final_truncated):
    ticks = [([1.0], [3], [False]), ([1.0], [3], [False]),
             ([1.0], [final_lives], [final_truncated])]
    agent = _make_agent(num_envs=1, ticks=ticks, max_train_frames=3, metrics_dir=tmp_path)

    agent.train()

    assert agent.rewards == [3.0]
    rows = _episode_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["episode_num"] == "1"
    assert float(rows[0]["episode_reward"]) == 3.0


@pytest.mark.parametrize("final_lives, final_truncated", [(0, False), (3, True)])
def test_only_finished_env_is_recorded_on_final_vector_tick(tmp_path, final_lives, final_truncated):
    ticks = [
        ([1.0, 2.0], [3, 3], [False, False]),
        ([1.0, 2.0], [final_lives, 3], [final_truncated, False]),
    ]
    agent = _make_agent(num_envs=2, ticks=ticks, max_train_frames=4, metrics_dir=tmp_path)

    agent.train()

    assert agent.rewards == [2.0]
    rows = _episode_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["episode_num"] == "1"
    assert float(rows[0]["episode_reward"]) == 2.0


def test_checkpoint_during_ongoing_game_does_not_record_episode(tmp_path):
    ticks = [([1.0], [3], [False])] * 2
    agent = _make_agent(num_envs=1, ticks=ticks, max_train_frames=2, metrics_dir=tmp_path)
    agent.checkpoint_path = str(tmp_path / "checkpoint")
    agent.checkpoint_freq = agent.replay_checkpoint_freq = 1
    saved_rewards = []

    def save(path, write_replay_memory):
        saved_rewards.append((list(agent.rewards), _episode_rows(tmp_path)))

    agent.save = save
    agent.train()

    assert saved_rewards == [([], []), ([], [])]
    assert agent.rewards == []
    assert _episode_rows(tmp_path) == []
