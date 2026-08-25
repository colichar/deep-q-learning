"""
Unit test for SpaceInvaderAgent.close() (issue #29 review follow-up): must close both
the vectorized training env and the standalone eval env, so AsyncVectorEnv's subprocess
workers (num_envs > 1) don't leak.
"""
from src.agent.agent import SpaceInvaderAgent


class FakeEnv:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_close_closes_both_vec_env_and_my_env():
    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.vec_env = FakeEnv()
    agent.my_env = FakeEnv()

    agent.close()

    assert agent.vec_env.closed
    assert agent.my_env.closed
