"""Action repeat runs inside each vector worker without crossing episode boundaries."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from src.utils.preprocessor import Preprocessor
from src.utils.space_invaders import ActionRepeatEnv


class ScriptedEnv(gym.Env):
    def __init__(self, base, lengths, truncate=False):
        self.base = base
        self.lengths = lengths
        self.truncate = truncate
        self.observation_space = gym.spaces.Box(0, 255, shape=(2,), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(3)
        self.episode = -1
        self.t = 0

    def _obs(self):
        value = self.base + 100 * self.episode + self.t
        return np.array([value, value + (10 if self.t % 2 else 0)], dtype=np.uint8)

    def reset(self, seed=None, options=None):
        self.episode += 1
        self.t = 0
        return self._obs(), {"lives": 3}

    def step(self, action):
        self.t += 1
        obs = self._obs()
        done = self.t >= self.lengths[self.episode]
        return obs, float(obs[0]), done and not self.truncate, done and self.truncate, {"lives": 3}


def _make_vec_env(vec_type, *specs):
    env = vec_type([
        lambda base=b, lengths=l, truncate=t: ActionRepeatEnv(ScriptedEnv(base, lengths, truncate))
        for b, l, t in specs
    ])
    env.reset(seed=0)
    return env


@pytest.mark.parametrize("vec_type", [SyncVectorEnv, AsyncVectorEnv])
def test_staggered_ends_stop_each_worker_without_crossing_games(vec_type):
    env = _make_vec_env(vec_type, (0, [1, 50], False), (10, [3, 50], True), (20, [50], False))
    try:
        obs, rewards, terminated, truncated, info = Preprocessor().step_with_skip_vec(env, [0, 0, 0])
        assert obs.ravel().tolist() == [1, 11, 13, 23, 24, 33]
        assert rewards.tolist() == [1, 11 + 12 + 13, 21 + 22 + 23 + 24]
        assert terminated.tolist() == [True, False, False]
        assert truncated.tolist() == [False, True, False]
        assert info["lives"].tolist() == [3, 3, 3]

        reset_obs, reset_info = env.reset(options={"reset_mask": np.array([True, True, False])})
        assert reset_obs.tolist() == [[100, 100], [110, 110], [24, 33]]
        assert reset_info["_lives"].tolist() == [True, True, False]

        obs, rewards, terminated, truncated, _ = Preprocessor().step_with_skip_vec(env, [0, 0, 0])
        assert obs.ravel().tolist() == [104, 113, 114, 123, 28, 37]
        assert rewards.tolist() == [101 + 102 + 103 + 104, 111 + 112 + 113 + 114, 25 + 26 + 27 + 28]
        assert not terminated.any() and not truncated.any()
    finally:
        env.close()


def test_end_on_first_frame_pools_with_itself():
    env = _make_vec_env(SyncVectorEnv, (0, [1], False), (10, [50], False))
    try:
        obs, rewards, terminated, _, _ = Preprocessor().step_with_skip_vec(env, [0, 0])
        assert obs.tolist() == [[1, 11], [14, 23]]
        assert rewards.tolist() == [1, 11 + 12 + 13 + 14]
        assert terminated.tolist() == [True, False]
    finally:
        env.close()


def test_single_env_and_one_worker_agree():
    single = ScriptedEnv(0, [6])
    single.reset()
    vector = _make_vec_env(SyncVectorEnv, (0, [6], False))
    try:
        for _ in range(2):
            expected = Preprocessor().step_with_skip(single, 0)
            actual = Preprocessor().step_with_skip_vec(vector, [0])
            assert actual[0][0].tolist() == expected[0].tolist()
            assert actual[1].tolist() == [expected[1]]
            assert actual[2].tolist() == [expected[2]]
            assert actual[3].tolist() == [expected[3]]
    finally:
        vector.close()
