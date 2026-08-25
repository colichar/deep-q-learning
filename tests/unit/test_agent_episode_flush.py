"""
Unit tests for SpaceInvaderAgent.train()'s end-of-run episode bookkeeping (issue #29).

The pre-vectorization single-env loop always recorded the current episode_reward once the
frame budget ran out, even mid-episode (an inner `while alive:` loop broken by the frame
budget, not by the episode actually ending, still fell through to the unconditional
`self.rewards.append(episode_reward)` afterwards) - the vectorized rewrite has to preserve
that for num_envs=1, and avoid double-recording an env whose episode happens to end on the
very last tick processed.
"""
import numpy as np
import torch

from src.agent.agent import SpaceInvaderAgent


class FakePreprocessor:
    """Feeds a prescripted (reward, lives) sequence per env, one tick at a time."""

    def __init__(self, num_envs, ticks):
        # ticks: list of (rewards, lives) pairs, each an array of length num_envs.
        self.num_envs = num_envs
        self.ticks = iter(ticks)

    def initialize_state_vec(self, vec_env):
        state = torch.zeros(self.num_envs, 4, 84, 84, dtype=torch.uint8)
        return state, {"lives": np.full(self.num_envs, 3)}

    def step_with_skip_vec(self, vec_env, actions):
        rewards, lives = next(self.ticks)
        obs = np.zeros((self.num_envs, 210, 160, 3), dtype=np.uint8)
        return obs, np.array(rewards, dtype=float), None, None, {"lives": np.array(lives)}

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


def _make_agent(num_envs, ticks, max_train_frames):
    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.num_envs = num_envs
    agent.Preprocessor = FakePreprocessor(num_envs, ticks)
    agent.vec_env = None
    agent.ReplayMemory = FakeReplayMemory()
    agent.ExploreVsExploit = FakeExploreVsExploit(num_envs)
    agent.metrics_dir = None
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
    return agent


def test_leftover_episode_flushed_once_when_budget_runs_out_mid_episode():
    # 3 ticks, lives stay at 3 throughout - the episode never naturally ends.
    ticks = [([1.0], [3]), ([1.0], [3]), ([1.0], [3])]
    agent = _make_agent(num_envs=1, ticks=ticks, max_train_frames=3)

    agent.train()

    assert agent.rewards == [3.0]


def test_natural_completion_on_the_final_tick_is_not_double_recorded():
    # 3 ticks, lives hit 0 exactly on the last tick processed.
    ticks = [([1.0], [3]), ([1.0], [3]), ([1.0], [0])]
    agent = _make_agent(num_envs=1, ticks=ticks, max_train_frames=3)

    agent.train()

    assert agent.rewards == [3.0]


def test_leftover_flush_is_independent_per_env():
    # env 0 finishes naturally on the last tick; env 1 never finishes.
    ticks = [
        (np.array([1.0, 2.0]), np.array([3, 3])),
        (np.array([1.0, 2.0]), np.array([0, 3])),
    ]
    agent = _make_agent(num_envs=2, ticks=ticks, max_train_frames=4)

    agent.train()

    assert sorted(agent.rewards) == [2.0, 4.0]
