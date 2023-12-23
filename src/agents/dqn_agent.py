from ..utils.preprocessing import Preprocessor
from ..utils.replay_memory import ReplayMemory
from ..models.cnn_model import CNNModel

import gym

from numpy import uint8, mean, random
from tensorflow import constant, where, GradientTape, reduce_max, expand_dims

from skimage.transform import resize
from PIL import Image

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
        average_loss_freq = 400,
        discount = 0.99,
        eval_episodes = 10
    ):
        self.my_env = gym.make("ALE/SpaceInvaders-v5", frameskip=5, render_mode="rgb_array")

        self.batch_size = batch_size
        self.max_train_frames = max_train_frames
        self.memory_warmup = memory_warmup
        self.update_main_freq = update_main_freq
        self.update_target_freq = update_target_freq
        self.average_loss_freq = average_loss_freq
        self.discount = constant(discount)
        self.eval_episodes = eval_episodes
        
        self.MainModel = CNNModel(self.my_env.action_space.n, learning_rate)
        self.TargetModel = CNNModel(self.my_env.action_space.n)
        self.TargetModel.set_weights(self.MainModel.get_weights())
        
        self.Preprocessor = Preprocessor()
        self.ReplayMemory = ReplayMemory(memory_size, self.batch_size)
        self.ExploreVsExploit = ExplorationVsExploitation(self.MainModel, self.my_env.action_space.n)
        
        self.losses = []
        self.averaged_losses = []
        self.rewards = []
        
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
        frame_num = 0
        
        while (frame_num < self.max_train_frames):
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
                    self.losses.append(loss)

                # perform weights update for target model
                if frame_num % self.update_target_freq == 0 and frame_num > self.memory_warmup:
                    self.TargetModel.set_weights(self.MainModel.get_weights())
                    print("Updating target model...")
                
                # averaging past losses
                if frame_num % self.average_loss_freq == 0 and frame_num > self.memory_warmup:
                    self.averaged_losses.append(mean(self.losses))
                    self.losses = []
                    print("Finished", frame_num, "frames. Loss:", self.averaged_losses[-1])

                curr_state = new_state
                curr_raw_obs = new_raw_obs
                frame_num += 1
                if frame_num >= self.max_train_frames:
                    break

            print("Episode finished. Reward:", episode_reward)
            self.rewards.append(episode_reward)
            
        return self.averaged_losses, self.rewards
    
    def export_as_gif(self, frames:list, name:str):
        resized_frames = [resize(frame, (420, 320, 3), preserve_range=True, order=0).astype(uint8) for frame in frames]

        images = [Image.fromarray(frame) for frame in resized_frames]

        images[0].save(name, save_all=True, append_images=images[1:], duration=100, loop=0)
    
    def evaluate(self):
        self.eval_rewards = []
        self.frames_for_gif = []
        num_of_ep = 1

        ## start outer loop of the number of episode we'll train the model for
        for episode in range(self.eval_episodes):

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
            return model_prediction
