"""
Unit tests for SpaceInvaderAgent.train()'s `just_reset` handling (issue #29 review
follow-up, Part B): once a sub-env's episode ends, the *next* tick's acting state for
that sub-env must be seeded from scratch (4 copies of the new frame) instead of
shift-appending onto the state stack of the episode that just ended, since gymnasium's
NEXT_STEP auto-reset means that tick's raw obs already belongs to the new episode.
"""
import numpy as np
import torch

from src.agent.agent import SpaceInvaderAgent


class FakePreprocessor:
    """Feeds a prescripted (reward, lives) sequence per env, one tick at a time, and does
    a genuine shift-append in new_state_vec (each tick's new frame is a distinct constant
    value), so a state stack's contents reveal which ticks' frames it was built from."""

    def __init__(self, num_envs, ticks):
        self.num_envs = num_envs
        self.ticks = iter(ticks)
        self.tick_idx = 0

    def initialize_state_vec(self, vec_env):
        state = torch.zeros(self.num_envs, 4, 84, 84, dtype=torch.uint8)
        return state, {"lives": np.full(self.num_envs, 3)}

    def step_with_skip_vec(self, vec_env, actions):
        rewards, lives = next(self.ticks)
        obs = np.zeros((self.num_envs, 210, 160, 3), dtype=np.uint8)
        return obs, np.array(rewards, dtype=float), None, None, {"lives": np.array(lives)}

    def new_state_vec(self, new_raw_obs, curr_states):
        self.tick_idx += 1
        marker = self.tick_idx * 10
        new_frames = torch.full((self.num_envs, 84, 84), marker, dtype=torch.uint8)
        new_states = torch.cat([curr_states[:, 1:], new_frames.unsqueeze(1)], dim=1)
        return new_states, new_frames


class FakeReplayMemory:
    count = 0  # always below memory_warmup, so update/target/loss-averaging never fire

    def add_frames(self, *args, **kwargs):
        pass


class FakeExploreVsExploit:
    """Records the acting state it's called with, on every tick."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.seen_states = []

    def __call__(self, curr_states, frame_num):
        self.seen_states.append(curr_states.clone())
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


def test_acting_state_reseeded_the_tick_after_an_episode_ends():
    # tick1: alive; tick2: episode ends (lives hit 0); tick3: new episode (auto-reset);
    # tick4: still alive, so its acting-state input is tick3's already-overridden state.
    ticks = [
        ([0.0], [3]),
        ([0.0], [0]),
        ([0.0], [3]),
        ([0.0], [3]),
    ]
    agent = _make_agent(num_envs=1, ticks=ticks, max_train_frames=4)

    agent.train()

    seen = agent.ExploreVsExploit.seen_states
    assert len(seen) == 4
    # tick3's acting-state input still carries tick1/tick2's stale frames (10, 20) -
    # the override only takes effect starting the tick *after* just_reset is set.
    expected_tick3 = torch.stack([
        torch.zeros(84, 84, dtype=torch.uint8),
        torch.zeros(84, 84, dtype=torch.uint8),
        torch.full((84, 84), 10, dtype=torch.uint8),
        torch.full((84, 84), 20, dtype=torch.uint8),
    ])
    assert torch.equal(seen[2][0], expected_tick3)
    # tick4's acting-state input is tick3's output, reseeded to 4 copies of tick3's new
    # frame (marker 30) instead of a shift-append that would carry the marker-20 frame
    # from the episode that just ended.
    expected_tick4 = torch.full((1, 4, 84, 84), 30, dtype=torch.uint8)
    assert torch.equal(seen[3], expected_tick4)


def test_reset_override_is_independent_per_env():
    # env 0's episode ends on tick 1; env 1 stays alive throughout. The override for env
    # 0 is computed during tick 2 (the tick after just_reset is set), so it's only
    # observable as tick 3's acting-state input.
    ticks = [
        (np.array([0.0, 0.0]), np.array([0, 3])),
        (np.array([0.0, 0.0]), np.array([3, 3])),
        (np.array([0.0, 0.0]), np.array([3, 3])),
    ]
    # frame_num advances by num_envs=2 per tick (1, 3, 5, ...); budget 6 covers exactly 3 ticks.
    agent = _make_agent(num_envs=2, ticks=ticks, max_train_frames=6)

    agent.train()

    seen = agent.ExploreVsExploit.seen_states
    tick3_input = seen[2]
    # env 0 reseeded to 4 copies of tick 2's frame (marker 20), env 1 shift-appended
    # normally (still carrying tick1's marker-10 frame at position 2).
    assert torch.equal(tick3_input[0], torch.full((4, 84, 84), 20, dtype=torch.uint8))
    assert torch.equal(tick3_input[1, 2], torch.full((84, 84), 10, dtype=torch.uint8))
    assert torch.equal(tick3_input[1, 3], torch.full((84, 84), 20, dtype=torch.uint8))
