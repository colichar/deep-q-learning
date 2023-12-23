"""
  Work in progress:
  
  Migrating the DQN model from jupyter notebook to script.
"""
import gym


# initialize the gym env
my_env = gym.make("ALE/SpaceInvaders-v5", frameskip=5, render_mode="rgb_array")
obs_shape = my_env.observation_space.shape
num_actions = my_env.action_space.n

# parameters
memory_size = 10**4                   # size of replay memory
max_frames = 5 * 10**4                # number of max frames per epoch
minibatch_size = 32                   # minibatch size
update_target_freq = 0.25 * 10**4     # update target model every 2500 frames
discount = 0.99                       # discount for the cummulative reward
update_main_freq = 4                  # update main model every 4 frames
average_loss_freq = 400               # average the loss of last 400 frames
learning_rate = 0.001                 # learning rate of the model
memory_warm_up = 10**4                # go first 10000 frames without learning

if __name__ == "__main__":
    print('Hello World!')