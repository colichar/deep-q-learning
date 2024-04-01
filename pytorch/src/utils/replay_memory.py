from torch.utils.data import Dataset
from numpy.random import choice as random_choice
from numpy import array as np_array
from torch import tensor, stack
import pickle

class ReplayMemory(Dataset):
    """Stores the experience (frames) of the agent so they can be replayed for training."""
    def __init__(self, max_samples, batch_size):
        self.max_samples = max_samples
        self.batch_size = batch_size

        self.idx = 0

        self.buffer_transitions = []

    def __len__(self):
        return len(self.buffer_transitions)

    def __getitem__(self, idx):
        return self.buffer_transitions[idx]
    
    def add_transition(self, transition):
        """
        Adds new transition to replay memory.
        
        Parameters:
        - transition (tuple): The path where the data should be saved.
        """

        if len(self.buffer_transitions) < self.max_samples:
            self.buffer_transitions.append(transition)
        else:
            self.idx %= self.max_samples
            self.buffer_transitions[self.idx] = transition
            self.idx += 1

    
    def get_batch(self):
        """
        Creates a randomly picked training batch from memory.
        """
        indices = random_choice(len(self.buffer_transitions), self.batch_size, replace=False)
        #sampled_data = [self.buffer_transitions[idx] for idx in indices]

        #return {
        #    'curr_state': [self.buffer_transitions[idx][0] for idx in indices],
        #    'new_state': [self.buffer_transitions[idx][1] for idx in indices],
        #    'curr_action': [self.buffer_transitions[idx][2] for idx in indices],
        #    'reward': [self.buffer_transitions[idx][3] for idx in indices],
        #    'lives': [self.buffer_transitions[idx][4] for idx in indices],
        #}

        return (
            stack([self.buffer_transitions[idx][0] for idx in indices]),
            stack([self.buffer_transitions[idx][1] for idx in indices]),
            tensor([self.buffer_transitions[idx][2] for idx in indices]),
            tensor([self.buffer_transitions[idx][3] for idx in indices]),
            tensor([self.buffer_transitions[idx][4] for idx in indices])
        )

    def save_replay_memory(self, path):
        """
        Saves the agents replay memory to disk.
    
        Parameters:
        - path (str): The path where the data should be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folder '{path}' created.")
            
        with open(path + '/replay_memory', 'wb') as file:
            pickle.dump(self.buffer_transitions, file)

    def load_replay_memory(self, path):
        """
        Loads the agents replay memory from disk.
    
        Parameters:
        - path (str): The path frome where the data should be loaded.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder '{replay_memory_path}' does not exist.")
        
        with open(path + '/replay_memory', 'rb') as file:
            self.buffer_transitions = pickle.load(file)