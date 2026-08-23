"""
Unit tests for the single-frame ring-buffer replay memory
(see plans/replay-memory-single-frame-buffer.md, plans/replay-memory-save-load.md).
"""
import numpy as np
import pytest

from src.utils.replay_memory import ReplayMemory


def _fill_with_episodes(mem, capacity, num_writes, seed=0):
    """Writes num_writes random frames, tracking which episode each slot belongs to."""
    rng = np.random.default_rng(seed)
    episode_id = np.zeros(capacity, dtype=np.int64)
    current_episode = 0

    for _ in range(num_writes):
        frame = rng.integers(0, 256, size=(84, 84), dtype=np.uint8)
        terminal = bool(rng.random() < 0.02)
        write_idx = mem.idx
        mem.add_frame(frame, action=0, reward=0.0, terminal=terminal)
        episode_id[write_idx] = current_episode
        if terminal:
            current_episode += 1

    return episode_id


def _assert_no_boundary_violations(mem, episode_id, capacity):
    violations = []
    for idx in range(capacity):
        if not mem._valid_index(idx):
            continue
        # curr_state ends at idx - 1, next_state ends at idx (see get_batch) - the
        # joint window spans state_length + 1 frames, idx - state_length .. idx.
        window_episode_ids = {episode_id[(idx - k) % capacity] for k in range(mem.state_length + 1)}
        if len(window_episode_ids) != 1:
            violations.append((idx, window_episode_ids))

    assert not violations, (
        f"{len(violations)} valid indices span multiple episodes, e.g. {violations[:5]}"
    )


def test_memory_footprint():
    capacity = 1_000_000
    mem = ReplayMemory(capacity=capacity, batch_size=32)
    total_bytes = mem.frames.nbytes + mem.actions.nbytes + mem.rewards.nbytes + mem.terminal.nbytes

    # Paper's target is ~6.6GB for a single-frame uint8 buffer at this capacity;
    # this layout (frames + actions + rewards + terminal) lands close to that.
    assert total_bytes / 1e9 == pytest.approx(7.07, abs=0.1)


def test_episode_boundaries():
    capacity = 500
    mem = ReplayMemory(capacity=capacity, batch_size=32)
    episode_id = _fill_with_episodes(mem, capacity, num_writes=3000)  # write past capacity to exercise wraparound

    # Exercise get_batch() itself (checks sampling doesn't hang or crash at scale).
    for _ in range(50):
        mem.get_batch()

    _assert_no_boundary_violations(mem, episode_id, capacity)


def test_get_batch_aligns_states_with_their_action():
    """
    Regression test for the curr/next state off-by-one: actions[i] must describe the
    transition curr_state (ending at i - 1) -> next_state (ending at i), not
    (state ending at i) -> (state ending at i + 1).
    """
    capacity = 50
    mem = ReplayMemory(capacity=capacity, batch_size=16)
    for i in range(30):
        frame = np.full((84, 84), i, dtype=np.uint8)
        mem.add_frame(frame, action=i, reward=float(i), terminal=False)

    curr_states, next_states, actions, rewards, terminal = mem.get_batch()

    for b in range(mem.batch_size):
        action = int(actions[b])
        # frame values were set equal to their write index, so a frame's value
        # doubles as "which index produced it" - action i produced frame i.
        assert curr_states[b, -1, 0, 0].item() == action - 1
        assert next_states[b, -1, 0, 0].item() == action
        assert rewards[b].item() == action


def test_save_load_roundtrip(tmp_path):
    capacity = 500
    mem = ReplayMemory(capacity=capacity, batch_size=32)
    episode_id = _fill_with_episodes(mem, capacity, num_writes=3000)

    mem.save_replay_memory(str(tmp_path))

    loaded = ReplayMemory(capacity=capacity, batch_size=32)
    loaded.load_replay_memory(str(tmp_path))

    assert np.array_equal(loaded.frames, mem.frames)
    assert np.array_equal(loaded.actions, mem.actions)
    assert np.array_equal(loaded.rewards, mem.rewards)
    assert np.array_equal(loaded.terminal, mem.terminal)
    assert loaded.idx == mem.idx
    assert loaded.count == mem.count

    # Sampling from the reloaded buffer should behave identically to the original.
    for _ in range(50):
        loaded.get_batch()
    _assert_no_boundary_violations(loaded, episode_id, capacity)


def test_save_load_shape_mismatch(tmp_path):
    mem = ReplayMemory(capacity=500, batch_size=32)
    mem.add_frame(np.zeros((84, 84), dtype=np.uint8), action=0, reward=0.0, terminal=False)
    mem.save_replay_memory(str(tmp_path))

    mismatched = ReplayMemory(capacity=1000, batch_size=32)
    with pytest.raises(ValueError):
        mismatched.load_replay_memory(str(tmp_path))
