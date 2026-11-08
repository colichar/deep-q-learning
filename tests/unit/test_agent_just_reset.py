"""Reset completed vector workers before choosing their next action."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.vector import SyncVectorEnv

from src.agent.agent import SpaceInvaderAgent
from src.utils.space_invaders import ActionRepeatEnv


class ScriptedEnv(gym.Env):
    def __init__(self, base, done_at, truncate=False, life_loss_at=None):
        self.base = base
        self.done_at = done_at
        self.truncate = truncate
        self.life_loss_at = life_loss_at
        self.observation_space = gym.spaces.Box(0, 255, shape=(1,), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self.episode = -1
        self.t = 0
        self.actions = []

    def reset(self, *, seed=None, options=None):
        self.episode += 1
        self.t = 0
        return np.array([self.base + self.episode * 100], dtype=np.uint8), {"lives": 3}

    def step(self, action):
        self.actions.append((self.episode, self.t, int(action)))
        self.t += 1
        done = self.t == self.done_at and self.episode == 0
        lives = 2 if self.life_loss_at is not None and self.t >= self.life_loss_at else 3
        return (np.array([self.base + self.episode * 100 + self.t], dtype=np.uint8),
                1.0, done and not self.truncate, done and self.truncate, {"lives": lives})


class ScriptedPreprocessor:
    def preprocess_frame(self, observation):
        return torch.full((1, 84, 84), int(observation[0]), dtype=torch.uint8)

    def initialize_state_vec(self, vec_env):
        obs, info = vec_env.reset()
        frames = torch.stack([self.preprocess_frame(frame) for frame in obs])
        return frames.repeat(1, 4, 1, 1), info

    def step_with_skip_vec(self, vec_env, actions):
        return vec_env.step(actions)

    def new_state_vec(self, observations, old_states):
        frames = torch.stack([self.preprocess_frame(obs) for obs in observations])
        return torch.cat([old_states[:, 1:], frames], dim=1), frames.squeeze(1)


class RecordingReplay:
    count = 0

    def __init__(self):
        self.transitions = []

    def add_frames(self, frames, actions, rewards, terminals, boundaries):
        self.transitions.append((frames.copy(), np.array(actions).copy(), rewards.copy(),
                                 terminals.copy(), boundaries.copy()))
        self.count += len(actions)


class RecordingPolicy:
    def __init__(self):
        self.seen_states = []

    def __call__(self, states, frame_num):
        self.seen_states.append(states.clone())
        return (states[:, -1, 0, 0] >= 100).int().numpy()

    def get_epsilon(self, frame_num):
        return 0.5


def _make_agent(*specs, ticks=2):
    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.num_envs = len(specs)
    agent.vec_env = SyncVectorEnv([
        lambda base=b, done_at=d, truncate=t, life_loss_at=l:
            ActionRepeatEnv(ScriptedEnv(base, done_at, truncate, life_loss_at))
        for b, d, t, l in specs
    ])
    agent.Preprocessor = ScriptedPreprocessor()
    agent.ReplayMemory = RecordingReplay()
    agent.ExploreVsExploit = RecordingPolicy()
    agent.metrics_dir = agent.checkpoint_path = None
    agent.memory_warmup = 1_000_000
    agent.update_main_freq = agent.update_target_freq = 1_000_000
    agent.average_loss_freq = agent.log_freq = 1_000_000
    agent.start_frame_num = 0
    agent.max_train_frames = ticks * agent.num_envs
    agent.rewards = []
    agent.losses = []
    agent.frame_nums = []
    agent.averaged_losses = []
    agent._logged_reward_idx = 0
    agent.cumulative_wall_clock_seconds = 0.0
    agent._replay_restored_unmarked = False
    return agent


@pytest.mark.parametrize("truncate", [False, True])
@pytest.mark.parametrize("num_envs", [1, 2])
def test_reset_precedes_next_action_without_extra_transition(truncate, num_envs):
    specs = [(0, 2, truncate, None)]
    if num_envs == 2:
        specs.append((20, 100, False, 5))
    agent = _make_agent(*specs)
    try:
        agent.train()

        assert agent.start_frame_num == agent.ReplayMemory.count == 2 * num_envs
        assert len(agent.ReplayMemory.transitions) == 2
        first, second = agent.ReplayMemory.transitions
        assert first[0][0, 0, 0] == 2  # final observation, including on truncation
        assert first[3][0] == (not truncate)
        assert first[4][0]
        assert second[1][0] == 1  # first action in the new game uses its reset state
        assert agent.ExploreVsExploit.seen_states[1][0, :, 0, 0].tolist() == [100] * 4
        assert agent.vec_env.envs[0].env.actions[:3] == [(0, 0, 0), (0, 1, 0), (1, 0, 1)]
        assert agent.rewards[0] == 2.0

        if num_envs == 2:
            assert not first[3][1] and not first[4][1]
            assert second[3][1] and second[4][1]  # life loss, without a game reset
            assert second[1][1] == 0
            assert agent.vec_env.envs[1].env.episode == 0
            assert agent.ExploreVsExploit.seen_states[1][1, -1, 0, 0] == 24
    finally:
        agent.vec_env.close()
