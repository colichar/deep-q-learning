from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms.v2 import Resize
from torch import cat, tensor
from numpy import maximum, zeros
from numpy.random import randint


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

        total_reward = 0.0
        frames = []
        terminated = truncated = False
        info = {}

        for _ in range(skip):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            frames.append(obs)
            if terminated or truncated:
                break

        if len(frames) == 1:
            # episode ended on the very first frame of this group - nothing to
            # pair, so max() against itself is a no-op.
            frames.append(frames[0])

        maxed_obs = maximum(frames[-2], frames[-1])

        return maxed_obs, total_reward, terminated, truncated, info

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

        `info` is the vector env's info from the last sub-step taken, so its entries
        for a sub-env that ended mid-group already describe that sub-env's new
        episode - the returned masks, not `info`, mark the episode boundary.

        Two things are deliberately left for issue #29, which wires this into the
        training loop. First, `SpaceInvaderAgent.train` currently derives the replay
        memory's `terminal` flag purely from `info["lives"]` and never looks at
        `terminated`/`truncated`; porting that pattern as-is would read the *next*
        episode's `info` for a sub-env that ended mid-group, mis-tag terminal frames and
        let `ReplayMemory` sample across an episode boundary. Second, under `NEXT_STEP`
        auto-reset a new episode's first `skip - k` frames are consumed inside the group
        where the previous episode ended, so this method never returns a fresh episode's
        reset frame the way `initialize_state` does for the single-env path.
        TODO(#29): remove this note once both points above are handled in the vectorized
        training loop.
        """
        skip = self.frame_skip if skip is None else skip

        num_envs = vec_env.num_envs
        total_rewards = zeros(num_envs)
        terminated = zeros(num_envs, dtype=bool)
        truncated = zeros(num_envs, dtype=bool)
        done = zeros(num_envs, dtype=bool)

        prev_frames = last_frames = None
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

            total_rewards[live] += rewards[live]
            terminated |= step_terminated & live
            truncated |= step_truncated & live
            done = terminated | truncated

            if done.all():
                break

        maxed_obs = maximum(prev_frames, last_frames)

        return maxed_obs, total_rewards, terminated, truncated, info

    def initialize_state(self, env):
        """
        Initializes the first state of an episode with the first 4 flicker-
        reduced frames.

        Takes a randomized number of no-op frame-skip groups first, so each
        episode starts from a different point in the game's otherwise-fixed
        opening sequence instead of always the same frame.
        """
        env.reset()
        n_groups = randint(4, 31)

        maxed_frames = []
        info = {}
        for _ in range(n_groups):
            maxed_obs, _, _, _, info = self.step_with_skip(env, action=0)
            maxed_frames.append(maxed_obs)

        maxed_frames = maxed_frames[-4:]
        processed_frames = [self.preprocess_frame(frame) for frame in maxed_frames]

        return cat(processed_frames, axis=0), info

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
