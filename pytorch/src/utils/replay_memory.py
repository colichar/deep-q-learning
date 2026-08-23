import os

import numpy as np
from numpy import zeros, uint8, int64, float32, bool_, stack
from numpy.random import randint
from torch import from_numpy


class ReplayMemory:
    """
    Stores the experience (frames) of the agent so they can be replayed for training.

    Frames are stored once each in a circular buffer. A "state" is a sample of
    consecutive frames, reconstructed by indexing into the buffer, so overlapping
    states share the same underlying bytes instead of being duplicated.
    """

    def __init__(self, capacity, batch_size, frame_height=84, frame_width=84, state_length=4):
        self.capacity = capacity
        self.batch_size = batch_size
        self.state_length = state_length

        self.frames = zeros((capacity, frame_height, frame_width), dtype=uint8)
        self.actions = zeros(capacity, dtype=int64)
        self.rewards = zeros(capacity, dtype=float32)
        self.terminal = zeros(capacity, dtype=bool_)

        self.idx = 0
        self.count = 0

    def add_frame(self, frame, action, reward, terminal):
        """
        Adds a single new preprocessed frame and its metadata to the replay memory.

        Parameters:
        - frame: (84, 84) uint8 array, the newest frame only (not a stacked state).
        - action (int): action that was taken and produced this frame (i.e. the action
          taken from the *previous* state, not from the state this frame completes).
        - reward (float): reward received for that action.
        - terminal (bool): whether this frame ended the episode (lives == 0).
        """
        self.frames[self.idx] = frame
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.terminal[self.idx] = terminal

        self.idx = (self.idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def _get_state(self, index):
        """Gathers the state_length consecutive frames ending at `index`."""
        indices = [(index - offset) % self.capacity for offset in reversed(range(self.state_length))]
        return self.frames[indices]

    def _valid_index(self, index):
        """
        `index` names a stored (action, reward, terminal) triple, i.e. the transition
        curr_state -> next_state where next_state is the state ending at `index` and
        curr_state is the state ending at `index - 1` (see get_batch). It is samplable
        only if state_length frames of history exist behind it - enough to build both
        of those state windows - and none of those state_length preceding frames belong
        to a different episode than `index` itself. `index` itself is allowed to be
        terminal (it's the natural terminal transition), so it's excluded from that
        check.

        Before the buffer has wrapped, the oldest frame is at position 0. Once it has
        wrapped, the oldest frame is at `self.idx` (the next slot due to be overwritten)
        instead - the ring's chronological seam moves there, so history has to be
        measured as distance from `self.idx`, not from a fixed 0.
        """
        oldest = 0 if self.count < self.capacity else self.idx
        age_from_oldest = (index - oldest) % self.capacity
        if age_from_oldest < self.state_length:
            # not enough history behind index yet
            return False

        for offset in range(1, self.state_length + 1):
            if self.terminal[(index - offset) % self.capacity]:
                # window would mix frames from different episodes
                return False

        return True

    def get_batch(self):
        """
        Creates a randomly picked training batch from memory.

        For a sampled index `i`, actions[i]/rewards[i]/terminal[i] describe the
        transition that produced frame i, i.e. curr_state -> next_state where
        next_state is the state ending at i and curr_state is the state ending at
        i - 1 (see add_frame).
        """
        upper_bound = self.count if self.count < self.capacity else self.capacity
        indices = []
        while len(indices) < self.batch_size:
            candidate = int(randint(0, upper_bound))
            if self._valid_index(candidate):
                indices.append(candidate)

        curr_states = stack([self._get_state((i - 1) % self.capacity) for i in indices])
        next_states = stack([self._get_state(i) for i in indices])

        return (
            from_numpy(curr_states),
            from_numpy(next_states),
            from_numpy(self.actions[indices]),
            from_numpy(self.rewards[indices]),
            from_numpy(self.terminal[indices]),
        )

    def save_replay_memory(self, path):
        """
        Saves the buffer arrays and write position to `path/replay_memory.npz`.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        np.savez(
            os.path.join(path, "replay_memory.npz"),
            frames=self.frames,
            actions=self.actions,
            rewards=self.rewards,
            terminal=self.terminal,
            idx=self.idx,
            count=self.count,
        )

    def load_replay_memory(self, path):
        """
        Restores the buffer arrays and write position from `path/replay_memory.npz`.
        """
        file_path = os.path.join(path, "replay_memory.npz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No replay memory found at '{file_path}'.")

        data = np.load(file_path)
        if data["frames"].shape != self.frames.shape:
            raise ValueError(
                f"Loaded replay memory shape {data['frames'].shape} doesn't match "
                f"this buffer's configured shape {self.frames.shape}."
            )

        self.frames = data["frames"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.terminal = data["terminal"]
        self.idx = int(data["idx"])
        self.count = int(data["count"])
