import numpy as np

from src.utils.preprocessor import FrameSkipStepper, Preprocessor


class RawFrameEnv:
    def __init__(self, terminal_at=None):
        self.terminal_at = terminal_at
        self.steps = 0

    def step(self, action):
        self.steps += 1
        return np.array([self.steps, 10 - self.steps]), float(self.steps), self.steps == self.terminal_at, False, {"lives": 3}


def test_stepwise_action_repeat_matches_existing_skip_result():
    env = RawFrameEnv()
    stepper = FrameSkipStepper(env, action=2, frame_skip=4)

    raw_frames = []
    while not stepper.complete:
        raw_frames.append(stepper.advance()[0])

    obs, reward, terminated, truncated, info = stepper.result()
    assert [frame.tolist() for frame in raw_frames] == [[1, 9], [2, 8], [3, 7], [4, 6]]
    assert obs.tolist() == [4, 7]
    assert reward == 10
    assert not terminated and not truncated
    assert info == {"lives": 3}


def test_stepwise_action_repeat_preserves_early_terminal_semantics():
    env = RawFrameEnv(terminal_at=1)
    stepper = FrameSkipStepper(env, action=2, frame_skip=4)
    stepper.advance()

    assert stepper.complete
    obs, reward, terminated, truncated, _ = stepper.result()
    assert env.steps == 1
    assert obs.tolist() == [1, 9]
    assert reward == 1
    assert terminated and not truncated
