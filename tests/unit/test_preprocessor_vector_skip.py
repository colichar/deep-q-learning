"""
Unit tests for Preprocessor.step_with_skip_vec (GitHub issue #27).

These run against a real gymnasium SyncVectorEnv wrapping scripted sub-envs, so the
auto-reset behaviour under test is gymnasium's actual NEXT_STEP semantics rather than
a hand-rolled imitation of them: after a sub-env reports done, its next step() call
returns the reset observation of the following episode (reward 0, done False), and the
call after that is a real step of that new episode.

Observations encode their origin as `base + 100 * episode + t`, so a frame from the
next episode maxed into a terminal frame shows up as a value >= 100 above the base.
Rewards mirror the observation value for the same reason.
"""
import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

from src.utils.preprocessor import Preprocessor


class ScriptedEnv(gym.Env):
    """Emits obs/reward `base + 100 * episode + t`; episode e terminates after `lengths[e]` steps."""

    def __init__(self, base, lengths):
        self.base = base
        self.lengths = lengths
        self.observation_space = gym.spaces.Box(0, 255, shape=(1,), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(3)
        self.episode = -1
        self.t = 0

    def _obs(self):
        return np.array([self.base + 100 * self.episode + self.t], dtype=np.uint8)

    def reset(self, seed=None, options=None):
        self.episode += 1
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        obs = self._obs()
        return obs, float(obs[0]), self.t >= self.lengths[self.episode], False, {}


def _make_vec_env(*specs):
    vec_env = SyncVectorEnv([lambda base=b, lengths=l: ScriptedEnv(base, lengths) for b, l in specs])
    vec_env.reset(seed=0)
    return vec_env


def _actions(vec_env):
    return np.zeros(vec_env.num_envs, dtype=np.int64)


def test_maxes_last_two_frames_of_the_group_per_sub_env():
    vec_env = _make_vec_env((0, [50]), (10, [50]))
    obs, rewards, terminated, truncated, _ = Preprocessor().step_with_skip_vec(
        vec_env, _actions(vec_env)
    )

    assert obs.ravel().tolist() == [4, 14]
    assert rewards.tolist() == [1 + 2 + 3 + 4, 11 + 12 + 13 + 14]
    assert not terminated.any()
    assert not truncated.any()


def test_sub_env_terminating_mid_group_is_not_maxed_with_next_episode():
    # env 0 ends on the 2nd of 4 frames, so gymnasium hands back its reset frame (100)
    # and then a real frame of the next episode (101) for the rest of the group.
    vec_env = _make_vec_env((0, [2, 50]), (10, [50]))
    obs, rewards, terminated, truncated, _ = Preprocessor().step_with_skip_vec(
        vec_env, _actions(vec_env)
    )

    assert obs.ravel().tolist() == [2, 14]
    assert rewards.tolist() == [1 + 2, 11 + 12 + 13 + 14]
    assert terminated.tolist() == [True, False]
    assert not truncated.any()


def test_staggered_terminations_are_frozen_independently():
    vec_env = _make_vec_env((0, [1, 50]), (10, [3, 50]), (20, [50]))
    obs, rewards, terminated, truncated, _ = Preprocessor().step_with_skip_vec(
        vec_env, _actions(vec_env)
    )

    # env 0 ends on the very first frame of the group, so it maxes against itself.
    assert obs.ravel().tolist() == [1, 13, 24]
    assert rewards.tolist() == [1, 11 + 12 + 13, 21 + 22 + 23 + 24]
    assert terminated.tolist() == [True, True, False]
    assert not truncated.any()


def test_group_after_a_mid_group_termination_stays_inside_the_new_episode():
    vec_env = _make_vec_env((0, [2, 50]), (10, [50]))
    preprocessor = Preprocessor()
    preprocessor.step_with_skip_vec(vec_env, _actions(vec_env))

    obs, rewards, terminated, _, _ = preprocessor.step_with_skip_vec(vec_env, _actions(vec_env))

    # env 0's new episode already spent t=0 (reset frame) and t=1 inside the previous
    # group, so this group covers t=2..5 of it.
    assert obs.ravel().tolist() == [105, 18]
    assert rewards.tolist() == [102 + 103 + 104 + 105, 15 + 16 + 17 + 18]
    assert not terminated.any()


def test_single_env_path_matches_the_n1_vectorized_path():
    preprocessor = Preprocessor()
    single_env = ScriptedEnv(base=0, lengths=[6])
    single_env.reset()
    vec_env = _make_vec_env((0, [6]))

    # Two groups: the first plain, the second ending on the episode's terminal frame.
    for _ in range(2):
        single = preprocessor.step_with_skip(single_env, action=0)
        vector = preprocessor.step_with_skip_vec(vec_env, _actions(vec_env))

        assert vector[0].ravel().tolist() == single[0].ravel().tolist()
        assert vector[1].tolist() == [single[1]]
        assert vector[2].tolist() == [single[2]]
        assert vector[3].tolist() == [single[3]]
