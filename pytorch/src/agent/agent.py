from numpy import random

class ExplorationVsExploitation:
    """
    This class handles the epsilon-greedy strategy which will be used to determine whether
    we will choose an action to explore new possibilites or exploit the accumulated experience.
    """
    def __init__(self,
                 dqn_model:CNNModelPY,
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

        
        if random.random() < eps:
            # we explore
            return random.randint(self.n_actions)
        else:
            # we choose the action yielding the highest reward according to our main model
            model_prediction = self.dqn_model.best_action(curr_state.unsqueeze(0))
            model_prediction = int(model_prediction[0].numpy())
            return model_prediction

import gym
from torch import where, max, no_grad, tensor
from torchvision.transforms import Resize
from numpy import mean
import matplotlib.pyplot as plt
from PIL import Image

class SpaceInvaderAgent:
    def __init__(
        self,
        learning_rate = 0.001,
        memory_size = 10**4, # 100,
        memory_warmup = 0.5 * 10**4, # 20, 
        batch_size = 64, # 5
        max_train_frames = 0.6 * 10**4, # 400
        update_main_freq = 4,
        update_target_freq = 0.25 * 10**4, # 25,
        log_freq = 0.2 * 10**4,
        average_loss_freq = 400, #20
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
        self.discount = discount
        
        self.MainModel = CNNModelPY(self.my_env.action_space.n, learning_rate)
        self.TargetModel = CNNModelPY(self.my_env.action_space.n)
        self.TargetModel.set_weights(self.MainModel.get_weights())
        
        self.Preprocessor = Preprocessor()
        self.ReplayMemory = ReplayMemory(memory_size, self.batch_size)
        self.ExploreVsExploit = ExplorationVsExploitation(self.MainModel, self.my_env.action_space.n)
        
        self.losses = []

        self.frame_nums = []
        self.averaged_losses = []
        self.rewards = []
        
        self.eval_rewards = []
        self.frames_for_gif = []


    def update_step(self):
        minibatch = self.ReplayMemory.get_batch()
    
        # Extract individual components from minibatch
        curr_states, new_states, curr_actions, rewards, lives = minibatch
    
        # Create boolean mask to identify transitions with lives = 0
        terminal_mask = (lives == 0)
    
        # Calculate Q values for non-terminal transitions
        with no_grad():
            target_pred = self.TargetModel.best_reward(new_states)

        target_q = rewards + self.discount * target_pred
    
        # For terminal transitions, use rewards directly
        target_q = where(terminal_mask, rewards, target_q)
    
        predictions = self.MainModel(curr_states)
        selected_q_values = max(predictions, axis=1)[0]

        loss = self.MainModel.custom_huber_loss(target_q, selected_q_values)

        self.MainModel.optimizer.zero_grad()
        loss.backward()
        self.MainModel.optimizer.step()
    
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
                    self.losses.append(loss.item())


                # perform weights update for target model
                if frame_num % self.update_target_freq == 0 and frame_num > self.memory_warmup:
                    self.TargetModel.set_weights(self.MainModel.get_weights())
                    print("Updating target model...")
                
                # averaging past losses
                if frame_num % self.average_loss_freq == 0 and frame_num > self.memory_warmup:
                    self.frame_nums.append(frame_num)
                    self.averaged_losses.append(mean(self.losses))
                    
                    self.losses = []

                    if frame_num % self.log_freq == 0:
                        print("Finished", frame_num, "frames. Loss:", self.averaged_losses[-1])

                curr_state = new_state
                curr_raw_obs = new_raw_obs
                frame_num += 1
                if frame_num > self.max_train_frames + self.start_frame_num:
                    break

            print("Episode finished. Reward:", episode_reward)
            self.rewards.append(episode_reward)
    
    def evaluate(self, eval_episodes):
        self.eval_rewards = []
        self.frames_for_gif = []
        num_of_ep = 1

        with no_grad():
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
                self.export_as_gif(self.frames_for_gif, "eval_" + str(num_of_ep) + "_" + str(episode_reward) + ".gif")
                self.frames_for_gif = []
                num_of_ep += 1

        return self.eval_rewards

    def export_as_gif(self, frames:list, name:str):
        resized_frames = [self.resize_frame(frame) for frame in frames]
        
        images = [Image.fromarray(frame) for frame in resized_frames]

        images[0].save(name, save_all=True, append_images=images[1:], duration=100, loop=0)
        
    def resize_frame(self, frame):
        # Ensure frame has the correct shape (height, width, channels)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convert frame to uint8 if not already
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            # Resize frame to (420, 320, 3)
            image = Image.fromarray(frame)
            resize_transform = Resize((420, 320))
            resized_image = resize_transform(image)
            # Convert resized image back to numpy array
            resized_frame = np.array(resized_image)
            return resized_frame
        else:
            # Handle unexpected frame shape
            raise ValueError("Invalid frame shape or data type.")

    def plot_history(self):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
        ax1.set_title('Training losses')
        x_values = self.frame_nums
        y_values = self.averaged_losses
        ax1.plot(x_values, y_values)
        ax2.set_title('Training rewards')
        ax2.plot(self.rewards, 'o')
        fig.tight_layout()

    def save(self,
             path
            ):
        """
        Saves the agents replay memory, training history and model weights to disk.
    
        Parameters:
        - path (str): The path where the data should be saved.
        """
    
        print('Saving replay memory to disk...')
        self.save_replay_memory(path + '/replay_memory')
        print('Replay memory saved.')
        print('Saving model to disk...')
        self.save_model(path + '/model')
        print('Model saved.')
        print('Saving training history to disk...')
        self.save_train_history(path + '/history')
        print('Training history saved.')

    def save_replay_memory(self, path):
        # Save replay memory
        self.ReplayMemory.save_replay_memory(path)

    def save_model(self, path):
        # Save model weights
        self.MainModel.save_model(path)

    def save_train_history(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folder '{path}' created.")

        train_history = {
            "averaged_losses": self.averaged_losses,
            "frame_nums": self.frame_nums,
            "losses": self.losses,
            "rewards": self.rewards
        }

        with open(path + '/train_history', 'wb') as file:
            pickle.dump(train_history, file)

    def load(self,
             path
            ):
        """
        Loads the agents replay memory, training history and model weights to disk.
    
        Parameters:
        - path (str): The path where the data should be saved.
        """

        print('Loading replay memory from disk...')
        self.load_replay_memory(path + '/replay_memory')
        print('Replay memory loaded.')
        print('Loading model weights and training history from disk...')
        self.load_model(path + '/model')
        self.load_train_history(path + '/history')
        print('Model weights and training history loaded.')

    def load_replay_memory(self, path):
        self.ReplayMemory.load_replay_memory(path)

    def load_model(self, path):
        self.MainModel.load_model(path)
        self.TargetModel.set_weights(self.MainModel.get_weights())

    def load_train_history(self,path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder '{path}' does not exist.")

        train_history = {}
        with open(path + '/train_history', 'rb') as file:
            train_history = pickle.load(file)

        print(train_history.keys())

        self.losses = train_history["losses"]
        self.averaged_losses = train_history["averaged_losses"]
        self.frame_nums = train_history["frame_nums"]
        self.rewards = train_history["rewards"]

        self.start_frame_num = train_history["frame_nums"][-1]
