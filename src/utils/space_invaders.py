"""Shared ALE Space Invaders environment setup.

Keeping this in one place makes training, evaluation, and the playable demo use
the same Atari settings.
"""

import ale_py
import gymnasium as gym
from numpy import random


gym.register_envs(ale_py)


class NoopResetEnv(gym.Wrapper):
    """Randomize the opening with the same no-op reset used for training."""

    def __init__(self, env, noop_action=0):
        super().__init__(env)
        self.noop_action = noop_action

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        n_groups = random.randint(4, 31)
        for _ in range(n_groups):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


def make_space_invaders_env():
    """Create the raw-frame ALE environment expected by :class:`Preprocessor`."""
    return NoopResetEnv(gym.make(
        "ALE/SpaceInvaders-v5",
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode="rgb_array",
    ))
