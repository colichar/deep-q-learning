from torch.utils.data import Dataset
from typing import Union
from random import sample
from numpy.random import choice as random_choice
from numpy import array as np_array
from torch import tensor, stack
import pickle
import os


class ReplayMemoryFromDisk(Dataset):
    """Stores the experience (frames) of the agent to disk, so they can be replayed for training."""

    def __init__(self,
                 max_samples: int,
                 batch_size: int,
                 max_buffer_size: int,
                 path: str = './replay_memory'):

        self.max_samples = max_samples
        self.batch_size = batch_size

        self.idx = 0

        self.buffer = []
        self.max_buffer_size = max_buffer_size
        self.path = path

    def __getitem__(self,
                    idx: Union[int, list]):
        """
        Returns sample(s) from dataset and index(ices) idx.
        :param idx: Index or list of indices of the sample in Dataset.
        :return: Returns sample of dataset.
        """
        # Get a batch data from disk (clarify if one sample and then batch or get whole batch) try out and see what
        # happens
        # write Collate function to handle batches (it should return all parts of the transition in different
        # lists for the agent to be able to vectorize)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Folder '{self.path}' does not exist.")

        # See if you can improve performance of this line
        random_batch = sample(os.listdir(self.path), self.batch_size)

        batch = []

        for memory in random_batch:
            with open(f'{self.path}/{memory}', 'rb') as file:
                batch.append(pickle.load(file))

        return batch

    def __len__(self) -> int:
        """
        :return: Returns the length of the current buffer stored in RAM.
        """
        return len(self.buffer)

    def empty_buffer(self) -> None:
        """
        Empties the buffer.
        :return: None
        """
        self.buffer = []

    def save_memories(self) -> None:
        """
        Saves memories to disk at location specified by path.
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print(f"Folder '{self.path}' created.")

        for idx in range(self.idx - len(self.buffer), self.idx):
            with open(f'{self.path}/memory_{idx}', 'wb') as file:
                pickle.dump(self.buffer[idx], file)

        self.empty_buffer()

    def add_transition(self, transition: tuple) -> None:
        """
        Adds a transition to the buffer. If the buffer reaches its limit it empties the buffer and saves the memories
        to disk.
        :param transition:
        :return:
        """

        if len(self.buffer) == self.max_buffer_size:
            self.save_memories()

        self.idx %= self.max_samples
        self.buffer.append(transition)
        self.idx += 1



