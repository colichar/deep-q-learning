"""
Unit tests for Preprocessor.initialize_state_vec / new_state_vec (issue #29), which build
and advance the stacked acting state for N parallel envs.

Uses a real gymnasium SyncVectorEnv wrapping a scripted, ROM-free sub-env that emits
ALE-shaped (210, 160, 3) frames, so preprocess_frame's grayscale/crop/resize pipeline runs
unmodified - same convention as test_preprocessor_vector_skip.py.
"""
import numpy as np
import gymnasium as gym
import torch
from gymnasium.vector import SyncVectorEnv

from src.utils.preprocessor import Preprocessor


class ImageScriptedEnv(gym.Env):
    """Emits a constant-valued (210, 160, 3) frame per step, value = base + t."""

    def __init__(self, base):
        self.base = base
        self.observation_space = gym.spaces.Box(0, 255, shape=(210, 160, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(3)
        self.t = 0

    def _obs(self):
        return np.full((210, 160, 3), min(self.base + self.t, 255), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        self.t = 0
        return self._obs(), {"lives": 3}

    def step(self, action):
        self.t += 1
        return self._obs(), 0.0, False, False, {"lives": 3}


def test_initialize_state_vec_matches_single_env_path_at_n1():
    preprocessor = Preprocessor()
    single_env = ImageScriptedEnv(base=10)
    vec_env = SyncVectorEnv([lambda: ImageScriptedEnv(base=10)])

    single_state, single_info = preprocessor.initialize_state(single_env)
    vec_states, vec_info = preprocessor.initialize_state_vec(vec_env)

    assert vec_states.shape == (1, 4, 84, 84)
    assert torch.equal(vec_states[0], single_state)
    assert vec_info["lives"].tolist() == [single_info["lives"]]


def test_initialize_state_vec_is_4_copies_of_the_first_post_reset_frame():
    preprocessor = Preprocessor()
    vec_env = SyncVectorEnv([lambda base=b: ImageScriptedEnv(base=base) for b in (10, 50, 90)])

    states, info = preprocessor.initialize_state_vec(vec_env)

    assert states.shape == (3, 4, 84, 84)
    assert states.dtype == torch.uint8
    assert info["lives"].tolist() == [3, 3, 3]
    for env_i in range(3):
        assert torch.equal(states[env_i, 0], states[env_i, 1])
        assert torch.equal(states[env_i, 0], states[env_i, 2])
        assert torch.equal(states[env_i, 0], states[env_i, 3])


def test_new_state_vec_matches_single_env_path_at_n1():
    preprocessor = Preprocessor()
    single_env = ImageScriptedEnv(base=10)
    vec_env = SyncVectorEnv([lambda: ImageScriptedEnv(base=10)])

    single_state, _ = preprocessor.initialize_state(single_env)
    vec_states, _ = preprocessor.initialize_state_vec(vec_env)

    new_raw_obs = np.full((1, 210, 160, 3), 200, dtype=np.uint8)
    single_new_state, single_new_frame = preprocessor.new_state(new_raw_obs[0], single_state)
    vec_new_states, vec_new_frames = preprocessor.new_state_vec(new_raw_obs, vec_states)

    assert torch.equal(vec_new_states[0], single_new_state)
    assert torch.equal(vec_new_frames[0], single_new_frame)


def test_new_state_vec_shifts_and_appends_per_env():
    preprocessor = Preprocessor()
    old_states = torch.arange(2 * 4 * 84 * 84, dtype=torch.uint8).reshape(2, 4, 84, 84)
    new_raw_obs = np.stack([
        np.full((210, 160, 3), 111, dtype=np.uint8),
        np.full((210, 160, 3), 222, dtype=np.uint8),
    ])

    new_states, new_frames = preprocessor.new_state_vec(new_raw_obs, old_states)

    assert new_states.shape == (2, 4, 84, 84)
    assert torch.equal(new_states[:, :3], old_states[:, 1:])
    assert torch.equal(new_states[:, 3], new_frames)
