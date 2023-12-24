from tensorflow import concat, cast, image, uint8
from numpy import array

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
        
        return concat(processed_frames, axis=-1), raw_frames[-1]
        
        
    def encode_frames(self,
                      new_raw_obs,
                      old_raw_obs,
                     ):
        
        """Encodes two consecutive frames in such a manner to remove the flickering of projectiles."""

        return array([old_raw_obs, new_raw_obs]).max(axis=0)
    
    def preprocess_frame(self,
                         new_raw_obs,
                         old_raw_obs,
                        ):
        """
        Preprocesses one frame for the model.
        """

        processed_fr = self.encode_frames(old_raw_obs, new_raw_obs)
        processed_fr = image.rgb_to_grayscale(processed_fr)
        processed_fr = image.crop_to_bounding_box(processed_fr, 34, 0, 160, 160)
        processed_fr = image.resize(processed_fr, [self.height, self.width], method='bilinear')
        processed_fr = cast(processed_fr, uint8)

        return processed_fr

    
    def new_state(self,
                  new_raw_obs,
                  old_raw_obs,
                  old_state
                 ):
        """
        Creates a news state from an old state and a new raw frame.
        """
        processed_fr = self.preprocess_frame(new_raw_obs, old_raw_obs)
        
        return concat([old_state[::, ::, 1:], processed_fr], axis=-1)