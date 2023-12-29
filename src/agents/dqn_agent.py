from ..utils.preprocessing import Preprocessor
from ..utils.replay_memory import ReplayMemory
from ..models.cnn_model import CNNModel

import gym

from numpy import uint8, random
from tensorflow import constant, where, GradientTape, reduce_max, reduce_mean, expand_dims, TensorArray, float32, stack
from tensorflow.io import write_file, serialize_tensor, parse_tensor, read_file

from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt

import os

class SpaceInvaderAgent:
    def __init__(
        self,
        learning_rate = 0.001,
        memory_size = 10**4,
        memory_warmup = 0.5 * 10**4,
        batch_size = 32,
        max_train_frames = 5 * 10**4,
        update_main_freq = 4,
        update_target_freq = 0.25 * 10**4,
        log_freq = 0.2 * 10**4,
        average_loss_freq = 400,
        discount = 0.99
    ):
        self.my_env = gym.make("ALE/SpaceInvaders-v5", frameskip=5, render_mode="rgb_array")

        self.start_frame_num = 0

        self.batch_size = batch_size
        self.max_train_frames = max_train_frames
        self.memory_warmup = memory_warmup
        self.update_main_freq = update_main_freq
        self.update_target_freq = update_target_freq
        self.average_loss_freq = average_loss_freq
        self.log_freq = log_freq
        self.discount = constant(discount)
        
        self.MainModel = CNNModel(self.my_env.action_space.n, learning_rate)
        self.TargetModel = CNNModel(self.my_env.action_space.n)
        self.TargetModel.set_weights(self.MainModel.get_weights())
        
        self.Preprocessor = Preprocessor()
        self.ReplayMemory = ReplayMemory(memory_size, self.batch_size)
        self.ExploreVsExploit = ExplorationVsExploitation(self.MainModel, self.my_env.action_space.n)
        
        self.losses = TensorArray(float32, size=0, dynamic_size=True)

        self.averaged_losses = TensorArray(float32, size=0, dynamic_size=True)
        self.rewards = TensorArray(float32, size=0, dynamic_size=True)
        
        self.eval_rewards = []
        self.frames_for_gif = []
        
    #@tf.function
    def update_step(self):
        minibatch = self.ReplayMemory._batch_random()
    
        # Extract individual components from minibatch
        curr_states, new_states, curr_actions, rewards, lives = minibatch
    
        # Create boolean mask to identify transitions with lives = 0
        terminal_mask = (lives == 0)
    
        # Calculate Q values for non-terminal transitions
        target_pred = self.TargetModel.best_reward(new_states)

        target_q = rewards + self.discount * target_pred
    
        # For terminal transitions, use rewards directly
        target_q = where(terminal_mask, rewards, target_q)
    
        with GradientTape() as tape:
            predictions = self.MainModel(curr_states)
            selected_q_values = reduce_max(predictions, axis=1)
            loss = self.MainModel.custom_huber_loss(target_q, selected_q_values)
    
        gradients = tape.gradient(loss, self.MainModel.trainable_variables)
        self.MainModel.optimizer.apply_gradients(zip(gradients, self.MainModel.trainable_variables))
    
        return loss

    
    def train(self):
        frame_num = self.start_frame_num + 1
        
        while (frame_num <= self.max_train_frames + self.start_frame_num):
            episode_reward = 0
            
            curr_state, curr_raw_obs = self.Preprocessor.initialize_state(self.my_env)
            alive = True
            
            while alive:
                # take action
                curr_action = self.ExploreVsExploit(curr_state, frame_num)

                new_raw_obs, reward, terminated, truncated, info = self.my_env.step(curr_action)

                alive = info["lives"] != 0

                episode_reward += reward

                reward = 1 if reward > 0 else -1 if reward < 0 else 0

                # create new sequence with new frame
                new_state = self.Preprocessor.new_state(new_raw_obs, curr_raw_obs, curr_state)

                # store new transition
                self.ReplayMemory.add_transition((curr_state, new_state, curr_action, reward, info["lives"]))

                # perform weights update for main model
                if frame_num % self.update_main_freq == 0 and frame_num > self.memory_warmup:
                    loss = self.update_step()
                    self.losses = self.losses.write(self.losses.size(), loss)


                # perform weights update for target model
                if frame_num % self.update_target_freq == 0 and frame_num > self.memory_warmup:
                    self.TargetModel.set_weights(self.MainModel.get_weights())
                    print("Updating target model...")
                
                # averaging past losses
                if frame_num % self.average_loss_freq == 0 and frame_num > self.memory_warmup:
                    current_averaged_loss = stack([frame_num, reduce_mean(self.losses.stack(), axis=0)])
                    self.averaged_losses = self.averaged_losses.write(
                        self.averaged_losses.size(),
                        current_averaged_loss
                    )
                    
                    self.losses = TensorArray(float32, size=0, dynamic_size=True)

                    if frame_num % self.log_freq == 0:
                        print("Finished", frame_num, "frames. Loss:", current_averaged_loss.numpy()[-1])

                curr_state = new_state
                curr_raw_obs = new_raw_obs
                frame_num += 1
                if frame_num > self.max_train_frames + self.start_frame_num:
                    break

            print("Episode finished. Reward:", episode_reward)
            self.rewards = self.rewards.write(
                        self.rewards.size(),
                        episode_reward
                    )
    
    def evaluate(self, eval_episodes):
        self.eval_rewards = []
        self.frames_for_gif = []
        num_of_ep = 1

        ## start outer loop of the number of episode we'll train the model for
        for episode in range(eval_episodes):

            episode_reward = 0
            alive = True

            ## initialise first sequence of new episode
            curr_state, curr_raw_obs = self.Preprocessor.initialize_state(self.my_env)

            while alive:

                ## choose an exploration/explotation action 
                curr_action = self.ExploreVsExploit(curr_state)

                ## take action
                new_raw_obs, reward, terminated, truncated, info = self.my_env.step(curr_action)
                
                alive = info["lives"] != 0

                self.frames_for_gif.append(new_raw_obs)

                episode_reward += reward

                ## create new sequence with new frame
                curr_state = self.Preprocessor.new_state(new_raw_obs, curr_raw_obs, curr_state)
                curr_raw_obs = new_raw_obs


            self.eval_rewards.append(episode_reward)
            self.export_as_gif(self.frames_for_gif, "eval_" + str(num_of_ep) + ".gif")
            self.frames_for_gif = []
            num_of_ep += 1

        return self.eval_rewards

    def export_as_gif(self, frames:list, name:str):
        resized_frames = [resize(frame, (420, 320, 3), preserve_range=True, order=0).astype(uint8) for frame in frames]

        images = [Image.fromarray(frame) for frame in resized_frames]

        images[0].save(name, save_all=True, append_images=images[1:], duration=100, loop=0)

    def plot_history(self):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
        ax1.set_title('Training losses')
        x_values = self.averaged_losses.stack().numpy()[::,0]
        y_values = self.averaged_losses.stack().numpy()[::,1]
        ax1.plot(x_values, y_values)
        ax2.set_title('Training rewards')
        ax2.plot(self.rewards.stack(), 'o')
        fig.tight_layout()

    def save(self,
             path,
             model_weights_prefix = 'mw',
             replay_memory_prefix = 'rm',
             train_history_prefix = 'th'
            ):
        """
        Saves the agents replay memory, training history and model weights to disk.
    
        Parameters:
        - path (str): The path where the data should be saved.
        """

        model_weights_path = path + '/model'
        replay_memory_path = path + '/replay_memory'
        train_history_path = model_weights_path + '/history'

        if not os.path.exists(replay_memory_path):
            os.makedirs(replay_memory_path)
            print(f"Folder '{replay_memory_path}' created.")
        
        if not os.path.exists(model_weights_path):
            os.makedirs(model_weights_path)
            print(f"Folder '{model_weights_path}' created.")

        if not os.path.exists(train_history_path):
            os.makedirs(train_history_path)
            print(f"Folder '{train_history_path}' created.")
    
        print('Saving replay memory to disk...')
        self.save_replay_memory(replay_memory_path + '/' + replay_memory_prefix)
        print('Replay memory saved.')
        print('Saving model weights and training history to disk...')
        self.save_model_weights(model_weights_path + '/' + model_weights_prefix)
        self.save_train_history(train_history_path + '/')
        print('Model weights and training history saved.')

    def save_replay_memory(self, path):
        # Save replay memory
        self.ReplayMemory.save_replay_memory(path)

    def save_model_weights(self, path):
        # Save model weights
        self.MainModel.save_weights(path)

    def save_train_history(self, path):
        # Save training history
        averaged_losses = self.averaged_losses.stack()
        losses = self.losses.stack()
        rewards = self.rewards.stack()

        write_file(path + '/losses.tf', serialize_tensor(losses))
        write_file(path + '/averaged_losses.tf', serialize_tensor(averaged_losses))
        write_file(path + '/rewards.tf', serialize_tensor(rewards))

    def load(self,
             path,
             model_weights_prefix = 'mw'
            ):
        """
        Loads the agents replay memory, training history and model weights to disk.
    
        Parameters:
        - path (str): The path where the data should be saved.
        """
        model_weights_path = path + '/model'
        replay_memory_path = path + '/replay_memory'
        train_history_path = model_weights_path + '/history'

        if not os.path.exists(replay_memory_path):
            raise FileNotFoundError(f"Folder '{replay_memory_path}' does not exist.")
        
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Folder '{model_weights_path}' does not exist.")

        if not os.path.exists(train_history_path):
            raise FileNotFoundError(f"Folder '{train_history_path}' does not exist.")

        print('Loading replay memory from disk...')
        self.load_replay_memory(replay_memory_path)
        print('Replay memory loaded.')
        print('Loading model weights and training history from disk...')
        self.load_model_weights(model_weights_path + '/' + model_weights_prefix)
        self.load_train_history(train_history_path)
        print('Model weights and training history loaded.')

    def load_replay_memory(self, path):
        self.ReplayMemory.load_replay_memory(path)

    def load_model_weights(self, path):
        self.MainModel.load_weights(path)
        self.TargetModel.set_weights(self.MainModel.get_weights())

    def load_train_history(self,path):
        losses = parse_tensor(read_file(path + '/losses.tf'), out_type=float32)
        averaged_losses = parse_tensor(read_file(path + '/averaged_losses.tf'), out_type=float32)
        rewards = parse_tensor(read_file(path + '/rewards.tf'), out_type=float32)

        self.losses = TensorArray(dtype=float32, size=losses.shape[0], dynamic_size=True)
        self.losses = self.losses.unstack(losses)
        self.averaged_losses = TensorArray(dtype=float32, size=averaged_losses.shape[0], dynamic_size=True)
        self.averaged_losses = self.averaged_losses.unstack(averaged_losses)
        self.rewards = TensorArray(dtype=float32, size=rewards.shape[0], dynamic_size=True)
        self.rewards = self.rewards.unstack(rewards)

        self.start_frame_num = int(averaged_losses[-1, 0].numpy())


class ExplorationVsExploitation:
    """
    This class handles the epsilon-greedy strategy which will be used to determine whether
    we will choose an action to explore new possibilites or exploit the accumulated experience.
    """
    def __init__(self,
                 dqn_model:CNNModel,
                 n_actions:int,
                 eps_initial:float=1.0,
                 eps_final:float=0.1,
                 start_fr:int=5000,
                 end_fr:int=1000000,
                 evaluation:bool=False):
        """
        Initiates an ExplorationVsExploitation object with a CNNModel object and hyperparameters.
        """
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.start_fr = start_fr
        self.end_fr = end_fr
        self.slope = (self.eps_initial - self.eps_final) / (self.start_fr - self.end_fr)
        self.intercept = eps_initial - self.slope * self.start_fr
        
        self.n_actions = n_actions
        
        self.dqn_model = dqn_model
        
        self.evaluation = evaluation
        
        
    def __call__ (self, curr_state, frame_num:int=0) -> int:
        """
        When the object is called, it will return an action to be performed by the agent.
        This action will either be an exploration of new possibilities or and exploitation
        of the accumulated experience of the agent.
        """
        if self.evaluation:
            eps = 0
        elif frame_num <= self.start_fr:
            eps = self.eps_initial
        elif self.start_fr < frame_num < self.end_fr:
            eps = self.slope * frame_num + self.intercept
        elif frame_num >= self.end_fr:
            eps = self.eps_final

        
        if random.rand() < eps:
            # we explore
            return random.randint(self.n_actions)
        else:
            # we choose the action yielding the highest reward according to our main model
            model_prediction = self.dqn_model.best_action(expand_dims(curr_state, axis=0))
            model_prediction = int(model_prediction.numpy())
            return model_prediction
