import os
from zipfile import BadZipFile

import numpy as np
from numpy import zeros, uint8, int64, float32, bool_, stack
from numpy.random import randint
from torch import cat, from_numpy


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
        self.boundary = zeros(capacity, dtype=bool_)

        self.idx = 0
        self.count = 0

    def add_frame(self, frame, action, reward, terminal, boundary=None):
        """
        Adds a single new preprocessed frame and its metadata to the replay memory.

        Parameters:
        - frame: (84, 84) uint8 array, the newest frame only (not a stacked state).
        - action (int): action that was taken and produced this frame (i.e. the action
          taken from the *previous* state, not from the state this frame completes).
        - reward (float): reward received for that action.
        - terminal (bool): whether to disable bootstrap for the transition into this
          frame, including learning terminals on life loss.
        - boundary (bool): whether history ends after this frame. Defaults to terminal
          to preserve life-loss boundaries for callers that only supply terminal.
        """
        self.frames[self.idx] = frame
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.terminal[self.idx] = terminal
        self.boundary[self.idx] = terminal if boundary is None else boundary

        self.idx = (self.idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def mark_restart_boundary(self):
        """
        Seals the newest stored frame as a history boundary, so frames appended by a
        fresh environment after a resume can never be sampled in the same state window
        as pre-restart frames. Written only when a checkpoint is restored from disk;
        an ordinary load/save round-trip is untouched. The transition into the seam
        frame itself stays samplable with its original bootstrap mask, since
        _valid_index exempts a boundary at `index`.

        No-op on an empty buffer - there is no last stored frame to seal.
        """
        if self.count > 0:
            self.boundary[(self.idx - 1) % self.capacity] = True

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
        of those state windows - and none of those state_length preceding frames has
        a boundary after it. A boundary after `index` itself is allowed: the transition
        into that frame still has intact history, regardless of its bootstrap mask.

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
            if self.boundary[(index - offset) % self.capacity]:
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

        # Write to a temp file and rename into place:
        # an interrupted in-place np.savez (Ctrl+C, kill, crash) leaves a truncated .npz
        # that fails with a bad-CRC error on the next load instead of a clean "file missing".
        final_path = os.path.join(path, "replay_memory.npz")
        tmp_path = os.path.join(path, "replay_memory.tmp.npz")
        np.savez(
            tmp_path,
            frames=self.frames,
            actions=self.actions,
            rewards=self.rewards,
            terminal=self.terminal,
            boundary=self.boundary,
            idx=self.idx,
            count=self.count,
        )
        os.replace(tmp_path, final_path)

    def load_replay_memory(self, path):
        """
        Restores the buffer arrays and write position from `path/replay_memory.npz`.
        """
        self._restore_snapshot(self._read_snapshot(path))

    def _read_snapshot(self, path):
        file_path = os.path.join(path, "replay_memory.npz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No replay memory found at '{file_path}'.")

        fields = ("frames", "actions", "rewards", "terminal", "boundary", "idx", "count")
        try:
            data = np.load(file_path, allow_pickle=False)
        except (OSError, BadZipFile, EOFError, ValueError) as error:
            raise ValueError(f"Cannot read replay memory '{file_path}': {error}") from error
        if not isinstance(data, np.lib.npyio.NpzFile):
            raise ValueError(f"Replay memory '{file_path}' is not an npz archive.")
        with data:
            missing = set(fields) - set(data.files)
            if missing:
                raise ValueError(f"Replay memory '{file_path}' is missing fields: {', '.join(sorted(missing))}.")
            try:
                snapshot = {name: data[name] for name in fields}
            except (OSError, BadZipFile, EOFError, ValueError) as error:
                raise ValueError(f"Cannot read replay memory '{file_path}': {error}") from error

        for name in ("frames", "actions", "rewards", "terminal", "boundary"):
            loaded = snapshot[name]
            expected = getattr(self, name)
            if loaded.shape != expected.shape or loaded.dtype != expected.dtype:
                raise ValueError(
                    f"Replay memory '{file_path}' {name} has shape {loaded.shape} and dtype {loaded.dtype}; "
                    f"expected shape {expected.shape} and dtype {expected.dtype}."
                )

        for name in ("idx", "count"):
            snapshot[name] = int(snapshot[name])
        return snapshot

    def _restore_snapshot(self, snapshot):
        for name, value in snapshot.items():
            setattr(self, name, value)


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

    @property
    def count(self):
        # Sub-buffers all receive exactly one add_frames() write per training tick, so
        # they grow in lockstep - this aggregate tracks total frames across all envs,
        # the same unit completed_transitions advances by per tick in the training loop.
        return sum(buffer.count for buffer in self.buffers)

    @staticmethod
    def _split_batch_sizes(batch_size, num_envs):
        """Distributes batch_size as evenly as possible across num_envs sub-buffers."""
        base, remainder = divmod(batch_size, num_envs)
        return [base + 1 if i < remainder else base for i in range(num_envs)]

    def add_frames(self, frames, actions, rewards, terminals, boundaries=None):
        """
        Adds one new frame per sub-env for this tick.

        Parameters (each of length num_envs, one entry per sub-env - see
        ReplayMemory.add_frame for the meaning of a single entry):
        - frames: (num_envs, 84, 84) uint8 array, the newest frame for each env.
        - actions: (num_envs,) ints.
        - rewards: (num_envs,) floats.
        - terminals: (num_envs,) bools.
        - boundaries: (num_envs,) bools, defaults to terminals.
        """
        if boundaries is None:
            boundaries = terminals
        for buffer, frame, action, reward, terminal, boundary in zip(
            self.buffers, frames, actions, rewards, terminals, boundaries
        ):
            buffer.add_frame(frame, action, reward, terminal, boundary)

    def get_batch(self):
        """
        Samples a batch from each sub-buffer (sizes from _split_batch_sizes) and
        concatenates them into a single batch of size batch_size, matching
        ReplayMemory.get_batch()'s return shape.
        """
        sub_batches = [buffer.get_batch() for buffer in self.buffers]
        return tuple(cat(tensors, dim=0) for tensors in zip(*sub_batches))

    def save_replay_memory(self, path):
        """Saves each sub-buffer under its own `path/env_<i>/` subfolder."""
        for i, buffer in enumerate(self.buffers):
            buffer.save_replay_memory(os.path.join(path, f"env_{i}"))

    def load_replay_memory(self, path):
        """Restores each sub-buffer from its own `path/env_<i>/` subfolder."""
        shard_paths = [os.path.join(path, f"env_{i}") for i in range(self.num_envs)]
        present = [os.path.isfile(os.path.join(shard, "replay_memory.npz")) for shard in shard_paths]
        if not any(present):
            for buffer in self.buffers:
                for name in ("frames", "actions", "rewards", "terminal", "boundary"):
                    getattr(buffer, name).fill(0)
                buffer.idx = buffer.count = 0
            raise FileNotFoundError(f"No replay memory snapshots found under '{path}'.")
        if not all(present):
            missing = [os.path.join(shard_paths[i], "replay_memory.npz") for i, exists in enumerate(present) if not exists]
            raise ValueError(f"Incomplete replay memory snapshot under '{path}'; missing: {', '.join(missing)}.")

        snapshots = [buffer._read_snapshot(shard) for buffer, shard in zip(self.buffers, shard_paths)]
        for buffer, snapshot in zip(self.buffers, snapshots):
            buffer._restore_snapshot(snapshot)

    def mark_restart_boundaries(self):
        """Seals the seam between restored replay history and fresh-environment frames
        in every nonempty sub-buffer. See ReplayMemory.mark_restart_boundary."""
        for buffer in self.buffers:
            buffer.mark_restart_boundary()
