from torch import cat

from src.utils.replay_memory import ReplayMemory


class VectorizedReplayMemory:
    """
    Wraps N independent ReplayMemory sub-buffers, one per parallel env, behind the
    same add/sample interface as a single ReplayMemory.

    Sub-buffers are fully independent - each has its own write head and episode
    boundaries, so a terminal frame in one env can never leak into another env's
    state windows. The aggregate `capacity` is split N ways rather than multiplied,
    so total memory footprint matches a single-env buffer of the same `capacity`.
    """

    def __init__(self, num_envs, capacity, batch_size, frame_height=84, frame_width=84, state_length=4):
        self.num_envs = num_envs
        self.batch_size = batch_size

        sub_capacity = capacity // num_envs
        sub_batch_sizes = self._split_batch_sizes(batch_size, num_envs)
        self.buffers = [
            ReplayMemory(sub_capacity, sub_batch_size, frame_height, frame_width, state_length)
            for sub_batch_size in sub_batch_sizes
        ]

    @staticmethod
    def _split_batch_sizes(batch_size, num_envs):
        """Distributes batch_size as evenly as possible across num_envs sub-buffers."""
        base, remainder = divmod(batch_size, num_envs)
        return [base + 1 if i < remainder else base for i in range(num_envs)]

    def add_frames(self, frames, actions, rewards, terminals):
        """
        Adds one new frame per sub-env for this tick.

        Parameters (each of length num_envs, one entry per sub-env - see
        ReplayMemory.add_frame for the meaning of a single entry):
        - frames: (num_envs, 84, 84) uint8 array, the newest frame for each env.
        - actions: (num_envs,) ints.
        - rewards: (num_envs,) floats.
        - terminals: (num_envs,) bools.
        """
        for buffer, frame, action, reward, terminal in zip(self.buffers, frames, actions, rewards, terminals):
            buffer.add_frame(frame, action, reward, terminal)

    def get_batch(self):
        """
        Samples a batch from each sub-buffer (sizes from _split_batch_sizes) and
        concatenates them into a single batch of size batch_size, matching
        ReplayMemory.get_batch()'s return shape.
        """
        sub_batches = [buffer.get_batch() for buffer in self.buffers]
        return tuple(cat(tensors, dim=0) for tensors in zip(*sub_batches))
