"""
Unit tests for agent.NoopResetEnv (issue #29 review follow-up), the OpenAI-Baselines-style
wrapper that performs a randomized no-op warmup inside reset() itself, so gymnasium's
NEXT_STEP auto-reset re-randomizes every episode's start during training, not just the
first.
"""
import gymnasium as gym

from src.agent.agent import NoopResetEnv


class ScriptedEnv(gym.Env):
    """Emits obs = step count since the last reset(); terminates after `episode_length` steps."""

    def __init__(self, episode_length=1000):
        self.episode_length = episode_length
        self.observation_space = gym.spaces.Box(0, 10 ** 6, shape=(1,), dtype=int)
        self.action_space = gym.spaces.Discrete(2)
        self.reset_count = 0
        self.t = 0

    def reset(self, seed=None, options=None):
        self.reset_count += 1
        self.t = 0
        return [self.t], {"reset_count": self.reset_count}

    def step(self, action):
        self.t += 1
        return [self.t], 0.0, self.t >= self.episode_length, False, {"reset_count": self.reset_count}


def test_reset_takes_randint_4_31_noop_steps(monkeypatch):
    monkeypatch.setattr("src.agent.agent.random.randint", lambda lo, hi: 12)

    env = NoopResetEnv(ScriptedEnv())
    obs, info = env.reset()

    assert obs == [12]


def test_reset_re_randomizes_the_noop_count_every_call(monkeypatch):
    counts = iter([4, 20])
    monkeypatch.setattr("src.agent.agent.random.randint", lambda lo, hi: next(counts))

    env = NoopResetEnv(ScriptedEnv())
    first_obs, _ = env.reset()
    second_obs, _ = env.reset()

    assert first_obs == [4]
    assert second_obs == [20]


def test_reset_recovers_from_termination_during_the_noop_warmup(monkeypatch):
    monkeypatch.setattr("src.agent.agent.random.randint", lambda lo, hi: 10)

    # episode_length=3 means the no-op warmup itself terminates the episode partway
    # through its 10 no-op steps; reset() must re-reset and keep going rather than crash.
    env = NoopResetEnv(ScriptedEnv(episode_length=3))
    obs, info = env.reset()

    # a step that terminates always has its obs overwritten by the following reset's,
    # so the returned obs can never be the terminal step count itself.
    assert obs[0] < 3
    assert info["reset_count"] > 1


def test_noop_action_is_used_for_the_warmup_steps():
    seen_actions = []

    class RecordingEnv(ScriptedEnv):
        def step(self, action):
            seen_actions.append(action)
            return super().step(action)

    env = NoopResetEnv(RecordingEnv(), noop_action=0)
    env.reset()

    assert set(seen_actions) == {0}
