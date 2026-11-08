from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms.v2 import Resize
from torch import cat, tensor, stack
from numpy import maximum


class FrameSkipStepper:
    """Advance one existing action-repeat group a raw ALE frame at a time.

    This is the step-wise counterpart to :meth:`Preprocessor.step_with_skip`.
    It lets a renderer show every emulator frame while retaining the training
    pipeline's four-frame action hold and last-two-frame max pooling.
    """

    def __init__(self, env, action, frame_skip):
        self.env = env
        self.action = action
        self.frame_skip = frame_skip
        self.total_reward = 0.0
        self.frames = []
        self.terminated = False
        self.truncated = False
        self.info = {}

    @property
    def complete(self):
        return len(self.frames) >= self.frame_skip or self.terminated or self.truncated

    def advance(self):
        """Take one raw frame and return its ALE observation and transition data."""
        obs, reward, self.terminated, self.truncated, self.info = self.env.step(self.action)
        self.total_reward += reward
        self.frames.append(obs)
        return obs, reward, self.terminated, self.truncated, self.info

    def result(self):
        """Return the same aggregate result as ``step_with_skip`` after completion."""
        frames = self.frames if len(self.frames) > 1 else self.frames * 2
        return maximum(frames[-2], frames[-1]), self.total_reward, self.terminated, self.truncated, self.info


class Preprocessor:
    """
    Takes care of frame-skipping and preprocessing frames of the game for the models.
    """

    def __init__(self, height: int = 84, width: int = 84, frame_skip: int = 4):
        self.height = height
        self.width = width
        self.frame_skip = frame_skip

    def step_with_skip(self, env, action, skip=None):
        """
        Repeats `action` for `skip` real ALE frames (or until the episode truly
        ends partway through), accumulating reward. Returns a single
        flicker-reduced observation built from the max of the last two real
        frames actually taken - the DeepMind papers' frame-skip + flicker-removal
        pairing, which needs the two frames being maxed to be adjacent. Requires
        the env itself to be created with `frameskip=1` (no internal ALE skipping).
        """
        skip = self.frame_skip if skip is None else skip

        stepper = FrameSkipStepper(env, action, skip)
        while not stepper.complete:
            stepper.advance()
        return stepper.result()

    def step_with_skip_vec(self, vec_env, actions):
        """Step workers that each own their action repeat and flicker removal."""
        return vec_env.step(actions)

    def initialize_state(self, env):
        """
        Initializes the first state of an episode as 4 copies of the first post-reset
        frame.
        """
        obs, info = env.reset()
        processed_fr = self.preprocess_frame(obs)

        return cat([processed_fr] * 4, axis=0), info

    def initialize_state_vec(self, vec_env):
        """
        Vectorized `initialize_state`: builds the first stacked state (num_envs, 4, H, W)
        for every sub-env at once, as 4 copies of each sub-env's first post-reset frame.
        """
        obs, info = vec_env.reset()
        processed = stack([self.preprocess_frame(frame) for frame in obs])

        return cat([processed] * 4, dim=1), info

    def crop_frame(self,
                   frame,
                   bounding_box=(34, 160, 0, 160)
                   ):
        """
        Crops frame to bounding box
        """
        y_min, height, x_min, width = bounding_box
        cropped_frame = frame[::, y_min:y_min + height, x_min:x_min + width]

        return cropped_frame

    def preprocess_frame(self, raw_obs):
        """
        Preprocesses one (already flicker-reduced) raw frame for the model.
        """

        processed_fr = tensor(raw_obs)
        processed_fr = rgb_to_grayscale(processed_fr.permute(2, 0, 1))
        processed_fr = self.crop_frame(processed_fr)
        processed_fr = Resize(size=(self.height, self.width))(processed_fr)

        return processed_fr

    def new_state(self, new_raw_obs, old_state):
        """
        Creates a new state from an old state and a new (already flicker-reduced)
        raw frame. Also returns the single new frame on its own, so it can be
        stored directly in the replay memory.
        """
        processed_fr = self.preprocess_frame(new_raw_obs)
        new_stacked_state = cat([old_state[1:, ::, ::], processed_fr], axis=0)

        return new_stacked_state, processed_fr.squeeze(0)

    def new_state_vec(self, new_raw_obs, old_states):
        """
        Vectorized `new_state`: `new_raw_obs` is (num_envs, H, W, C), `old_states` is
        (num_envs, 4, H, W). Also returns the new frames on their own (num_envs, H, W),
        so they can be stored directly in the replay memory.
        """
        processed = stack([self.preprocess_frame(frame) for frame in new_raw_obs])
        new_stacked_states = cat([old_states[:, 1:], processed], dim=1)

        return new_stacked_states, processed.squeeze(1)
