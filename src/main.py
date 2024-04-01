"""
  Work in progress:
  
  Migrating the DQN model from jupyter notebook to script.
"""
from tensorflow import constant
import sys
sys.path.append('/home/harisc/repos/deep-q-learning')
from src.agents.dqn_agent import SpaceInvaderAgent


# parameters
memory_size = constant(500)                   # size of replay memory
max_train_frames = constant(1000)                # number of max frames per epoch
batch_size = constant(32)                   # minibatch size
update_target_freq = 250     # update target model every 2500 frames
discount = 0.99                       # discount for the cummulative reward
update_main_freq = 4                  # update main model every 4 frames
average_loss_freq = 400               # average the loss of last 400 frames
learning_rate = 0.001                 # learning rate of the model
memory_warmup = 250                 # go first 10000 frames without learning
log_freq = 200

if __name__ == "__main__":
    my_agent = SpaceInvaderAgent(
        learning_rate = learning_rate,
        memory_size = memory_size,
        batch_size = batch_size,
        max_train_frames = max_train_frames,
        update_main_freq = update_main_freq,
        update_target_freq = update_target_freq,
        average_loss_freq = average_loss_freq,
        log_freq = log_freq,
        memory_warmup = memory_warmup,
        discount = discount
    )

    from time import time

    start = time()
    my_agent.train()
    end = time()
    print((end-start)/60, 'minutes elapsed')