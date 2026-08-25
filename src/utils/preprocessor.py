from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms.v2 import Resize
from torch import cat, tensor, stack
from numpy import maximum, zeros


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

    def step_with_skip_vec(self, vec_env, actions, skip=None):
        """
        Vectorized `step_with_skip`, for a gymnasium vector env in `NEXT_STEP`
        auto-reset mode. Repeats `actions` for `skip` real ALE frames and returns
        one flicker-reduced observation per sub-env, plus per-sub-env reward /
        terminated / truncated arrays.

        Each sub-env is frozen at the frame on which it reports done: its reward
        stops accumulating and its two maxed frames stay the last two of the episode
        that just ended. Under `NEXT_STEP` auto-reset the vector env keeps producing
        frames for that sub-env within the same group (first the reset observation,
        then real steps of the *next* episode), and maxing a terminal frame against
        one of those would silently splice two episodes together.

        `info` is the vector env's info from the last sub-step taken, with `lives`
        overwritten to be frozen the same way the frames are: for a sub-env that ended
        mid-group, `info["lives"]` reflects the episode that just ended, not the reset
        frames of the next one that gymnasium hands back for the rest of the group.
        """
        skip = self.frame_skip if skip is None else skip

        num_envs = vec_env.num_envs
        total_rewards = zeros(num_envs)
        terminated = zeros(num_envs, dtype=bool)
        truncated = zeros(num_envs, dtype=bool)
        done = zeros(num_envs, dtype=bool)

        prev_frames = last_frames = None
        frozen_lives = None
        info = {}

        for _ in range(skip):
            obs, rewards, step_terminated, step_truncated, info = vec_env.step(actions)
            live = ~done

            if last_frames is None:
                # first frame of the group has nothing to pair with yet, so a sub-env
                # ending here maxes against itself (same as the single-env path)
                prev_frames = obs.copy()
                last_frames = obs.copy()
            else:
                prev_frames[live] = last_frames[live]
                last_frames[live] = obs[live]

            if "lives" in info:
                if frozen_lives is None:
                    frozen_lives = info["lives"].copy()
                else:
                    frozen_lives[live] = info["lives"][live]

            total_rewards[live] += rewards[live]
            terminated |= step_terminated & live
            truncated |= step_truncated & live
            done = terminated | truncated

            if done.all():
                break

        maxed_obs = maximum(prev_frames, last_frames)

        if frozen_lives is not None:
            info = dict(info)
            info["lives"] = frozen_lives

        return maxed_obs, total_rewards, terminated, truncated, info

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
