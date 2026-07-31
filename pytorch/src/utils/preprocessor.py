from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms.v2 import Resize
from torch import cat, tensor
from numpy import array, maximum
from numpy.random import randint


class Preprocessor:
    """
    Takes care of preprocessing frames of the game for the models.
    """

    def __init__(self, height: int = 84, width: int = 84):
        self.height = height
        self.width = width

    def initialize_state(self, env):
        """
        Initializes the first state of an episode with the first 4 frames.

        Takes a randomized number of no-op actions first, so each episode starts
        from a different point in the game's otherwise-fixed opening sequence
        instead of always the same frame, then builds the initial 4-frame state
        from the last 5 resulting raw frames.
        """
        env.reset()
        n_noops = randint(5, 31)
        steps = [env.step(0) for _ in range(n_noops)]
        raw_frames = [step[0] for step in steps][-5:]
        info = steps[-1][4]
        processed_frames = [self.preprocess_frame(raw_frames[idx], raw_frames[idx + 1]) for idx in range(4)]

        return cat(processed_frames, axis=0), raw_frames[-1], info

    def encode_frames(self,
                      new_raw_obs,
                      old_raw_obs,
                      ):
        """Encodes two consecutive frames in such a manner to remove the flickering of projectiles."""

        return maximum(old_raw_obs, new_raw_obs)

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

    def preprocess_frame(self,
                         new_raw_obs,
                         old_raw_obs,
                         ):
        """
        Preprocesses one frame for the model.
        """

        processed_fr = tensor(self.encode_frames(old_raw_obs, new_raw_obs))
        processed_fr = rgb_to_grayscale(processed_fr.permute(2, 0, 1))
        processed_fr = self.crop_frame(processed_fr)
        processed_fr = Resize(size=(self.height, self.width))(processed_fr)

        return processed_fr

    def new_state(self,
                  new_raw_obs,
                  old_raw_obs,
                  old_state
                  ):
        """
        Creates a new state from an old state and a new raw frame. Also returns the
        single new frame on its own, so it can be stored directly in the replay memory.
        """
        processed_fr = self.preprocess_frame(new_raw_obs, old_raw_obs)
        new_stacked_state = cat([old_state[1:, ::, ::], processed_fr], axis=0)

        return new_stacked_state, processed_fr.squeeze(0)