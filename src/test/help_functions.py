import gym
import numpy as np
from ..utils.preprocessing import Preprocessor

## get some observations to test the preprocess function
## first 45 memory entries are before the game starts

def get_sample_frames(name="ALE/SpaceInvaders-v5", frameskip=5):

    memory_raw = []
    memory_processed = []
    memory_encoded = []

    some_preprocessor = Preprocessor()
    
    if frameskip:
        env = gym.make(name, frameskip=frameskip, render_mode="rgb_array")
    else:
        env = gym.make(name, render_mode="rgb_array")
        
    preprocessed, curr_raw_obs = some_preprocessor.initialize_state(env)
    memory_processed.append(preprocessed)
    memory_raw.append(curr_raw_obs)
    
    alive = True
    
    while alive:
        action = env.action_space.sample()
        new_raw_obs, reward, terminated, truncated, info = env.step(action)

        preprocessed = some_preprocessor.new_state(new_raw_obs, curr_raw_obs, memory_processed[-1])
        memory_processed.append(preprocessed)
        
        encoded = some_preprocessor.encode_frames(new_raw_obs, curr_raw_obs)
        memory_encoded.append(encoded)
        
        curr_raw_obs = new_raw_obs
        memory_raw.append(curr_raw_obs)
        
        if terminated:
            alive = False
            observation = env.reset()
    env.close()
    return memory_raw, memory_processed, memory_encoded