from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms.v2 import Resize
from torch import cat, tensor
from numpy import array, maximum

class Preprocessor:
    """
    Takes care of preprocessing frames of the game for the models.
    """
    
    def __init__(self, height:int=84, width:int=84):
        self.height = height
        self.width = width
        
    def initialize_state(self, env):
        """
        Initializes the first state of an episode with the first 4 frames.
        """
        env.reset()
        actions = [env.action_space.sample() for i in range(5)]
        raw_frames = [env.step(action)[0] for action in actions]
        processed_frames = [self.preprocess_frame(raw_frames[idx], raw_frames[idx+1]) for idx in range(4)]
        
        return cat(processed_frames, axis=0).float(), raw_frames[-1]
        
        
    def encode_frames(self,
                      new_raw_obs,
                      old_raw_obs,
                     ):
        
        """Encodes two consecutive frames in such a manner to remove the flickering of projectiles."""

        return maximum(old_raw_obs, new_raw_obs)

    def crop_frame(self,
                   frame,
                   bounding_box = (34, 160, 0, 160)
                  ):
        """
        Crops frame to bounding box
        """
        y_min, height, x_min, width  = bounding_box
        cropped_frame = frame[::, y_min:y_min+height, x_min:x_min+width]
        
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

        return processed_fr.float()

    
    def new_state(self,
                  new_raw_obs,
                  old_raw_obs,
                  old_state
                 ):
        """
        Creates a news state from an old state and a new raw frame.
        """
        processed_fr = self.preprocess_frame(new_raw_obs, old_raw_obs)
        
        return cat([old_state[1:, ::, ::], processed_fr], axis=0)