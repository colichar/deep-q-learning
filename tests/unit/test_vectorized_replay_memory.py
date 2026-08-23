"""
Unit tests for the vectorized replay memory wrapper (N independent ReplayMemory
sub-buffers, one per parallel env - see issue #26 / epic #25).
"""
import numpy as np
import pytest

from src.utils.replay_memory import VectorizedReplayMemory


def _fill_with_episodes(vmem, capacity_per_env, num_writes, seed=0):
    """
    Writes num_writes ticks (one frame per env each tick), tracking which episode
    each (env, slot) belongs to. Envs terminate independently and at different
    rates so sub-buffers exercise genuinely different episode boundaries.
    """
    rng = np.random.default_rng(seed)
    num_envs = vmem.num_envs
    episode_id = np.zeros((num_envs, capacity_per_env), dtype=np.int64)
    current_episode = np.zeros(num_envs, dtype=np.int64)

    for _ in range(num_writes):
        frames = rng.integers(0, 256, size=(num_envs, 84, 84), dtype=np.uint8)
        actions = np.zeros(num_envs, dtype=np.int64)
        rewards = np.zeros(num_envs, dtype=np.float32)
        # different terminal probability per env so boundaries don't line up across envs
        terminals = rng.random(num_envs) < (0.01 * (np.arange(num_envs) + 1))

        write_indices = [buffer.idx for buffer in vmem.buffers]
        vmem.add_frames(frames, actions, rewards, terminals)

        for env in range(num_envs):
            episode_id[env, write_indices[env]] = current_episode[env]
            if terminals[env]:
                current_episode[env] += 1

    return episode_id


def _assert_no_boundary_violations(buffer, episode_id_for_env, capacity):
    violations = []
    for idx in range(capacity):
        if not buffer._valid_index(idx):
            continue
        window_episode_ids = {
            episode_id_for_env[(idx - k) % capacity] for k in range(buffer.state_length + 1)
        }
        if len(window_episode_ids) != 1:
            violations.append((idx, window_episode_ids))

    assert not violations, (
        f"{len(violations)} valid indices span multiple episodes, e.g. {violations[:5]}"
    )


def test_capacity_split_across_subbuffers():
    num_envs = 4
    capacity = 1000
    vmem = VectorizedReplayMemory(num_envs=num_envs, capacity=capacity, batch_size=32)

    assert len(vmem.buffers) == num_envs
    for buffer in vmem.buffers:
        assert buffer.capacity == capacity // num_envs


def test_num_envs_one_keeps_full_capacity_and_batch_size():
    # num_envs=1 must be behaviorally unchanged from a single ReplayMemory (epic #25).
    vmem = VectorizedReplayMemory(num_envs=1, capacity=1000, batch_size=32)

    assert len(vmem.buffers) == 1
    assert vmem.buffers[0].capacity == 1000
    assert vmem.buffers[0].batch_size == 32


def test_batch_size_split_across_subbuffers():
    # 32 doesn't divide evenly by 5, so the remainder should land on the first envs.
    vmem = VectorizedReplayMemory(num_envs=5, capacity=1000, batch_size=32)
    sub_batch_sizes = [buffer.batch_size for buffer in vmem.buffers]

    assert sub_batch_sizes == [7, 7, 6, 6, 6]
    assert sum(sub_batch_sizes) == 32


def test_episode_boundaries_independent_per_subbuffer():
    num_envs = 3
    capacity_per_env = 200
    capacity = capacity_per_env * num_envs
    vmem = VectorizedReplayMemory(num_envs=num_envs, capacity=capacity, batch_size=32)

    episode_id = _fill_with_episodes(vmem, capacity_per_env, num_writes=1500)  # exercise wraparound

    for _ in range(50):
        vmem.get_batch()

    for env, buffer in enumerate(vmem.buffers):
        _assert_no_boundary_violations(buffer, episode_id[env], capacity_per_env)


def test_add_frames_routes_each_env_to_its_own_subbuffer():
    num_envs = 3
    vmem = VectorizedReplayMemory(num_envs=num_envs, capacity=300, batch_size=6)

    frames = np.stack([np.full((84, 84), env, dtype=np.uint8) for env in range(num_envs)])
    actions = np.array([10, 20, 30])
    rewards = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    terminals = np.array([False, False, False])

    vmem.add_frames(frames, actions, rewards, terminals)

    for env, buffer in enumerate(vmem.buffers):
        assert buffer.frames[0, 0, 0] == env
        assert buffer.actions[0] == actions[env]
        assert buffer.rewards[0] == pytest.approx(rewards[env])


def test_get_batch_correctness():
    num_envs = 4
    batch_size = 20
    vmem = VectorizedReplayMemory(num_envs=num_envs, capacity=800, batch_size=batch_size)

    rng = np.random.default_rng(1)
    for _ in range(500):
        frames = rng.integers(0, 256, size=(num_envs, 84, 84), dtype=np.uint8)
        actions = rng.integers(0, 6, size=num_envs)
        rewards = rng.random(num_envs).astype(np.float32)
        terminals = rng.random(num_envs) < 0.02
        vmem.add_frames(frames, actions, rewards, terminals)

    curr_states, next_states, actions, rewards, terminal = vmem.get_batch()

    assert curr_states.shape == (batch_size, 4, 84, 84)
    assert next_states.shape == (batch_size, 4, 84, 84)
    assert actions.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert terminal.shape == (batch_size,)
