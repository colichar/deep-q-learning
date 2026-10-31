"""
Integration tests for SpaceInvaderAgent: real gym/ALE env, real model, real filesystem.
Slower than the ReplayMemory unit tests - run with `uv run pytest -m integration`
or skip with `uv run pytest -m "not integration"`.
"""
import numpy as np
import pytest
from gymnasium.vector import AsyncVectorEnv

from src.agent.agent import SpaceInvaderAgent

pytestmark = pytest.mark.integration

AGENT_KWARGS = dict(
    memory_size=2000,
    memory_warmup=200,
    batch_size=16,
    max_train_frames=600,
    update_main_freq=4,
    update_target_freq=250,
    log_freq=200,
    average_loss_freq=100,
)


@pytest.mark.parametrize("double_dqn", [False, True])
def test_training_runs_end_to_end(double_dqn):
    algorithm_kwargs = {"double_dqn": True} if double_dqn else {}
    agent = SpaceInvaderAgent(**AGENT_KWARGS, **algorithm_kwargs)
    try:
        assert agent.double_dqn is double_dqn
        agent.train()

        assert len(agent.averaged_losses) > 0
        assert np.isfinite(agent.averaged_losses).all()
        assert len(agent.rewards) > 0
    finally:
        agent.close()


def test_vectorized_training_checkpoint_resume(tmp_path):
    agent = SpaceInvaderAgent(**AGENT_KWARGS, num_envs=2)
    try:
        assert isinstance(agent.vec_env, AsyncVectorEnv)
        assert agent.vec_env.num_envs == 2
        agent.train()

        assert len(agent.averaged_losses) > 0
        assert np.isfinite(agent.averaged_losses).all()
        assert len(agent.ReplayMemory.buffers) == 2
        assert all(buffer.count > 0 for buffer in agent.ReplayMemory.buffers)
        agent.save(str(tmp_path))
    finally:
        agent.close()

    resumed = SpaceInvaderAgent(**AGENT_KWARGS, num_envs=2)
    try:
        assert isinstance(resumed.vec_env, AsyncVectorEnv)
        assert resumed.vec_env.num_envs == 2
        resumed.load(str(tmp_path))

        assert resumed.start_frame_num == agent.start_frame_num == 600
        assert resumed.frame_nums == agent.frame_nums
        assert resumed.averaged_losses == agent.averaged_losses
        assert resumed.rewards == agent.rewards
        assert len(resumed.ReplayMemory.buffers) == 2
        for saved, loaded in zip(agent.ReplayMemory.buffers, resumed.ReplayMemory.buffers):
            for name in ("frames", "actions", "rewards", "terminal", "boundary"):
                np.testing.assert_array_equal(getattr(loaded, name), getattr(saved, name))
            assert loaded.idx == saved.idx
            assert loaded.count == saved.count

        resumed.train()

        assert resumed.start_frame_num == 1200
        assert len(resumed.averaged_losses) > len(agent.averaged_losses)
        assert np.isfinite(resumed.averaged_losses).all()
        assert resumed.frame_nums[-1] > agent.frame_nums[-1]
        assert len(resumed.rewards) > len(agent.rewards)
        for saved, continued in zip(agent.ReplayMemory.buffers, resumed.ReplayMemory.buffers):
            assert continued.count > saved.count
    finally:
        resumed.close()


@pytest.mark.parametrize("memory_warmup", [100, 18], ids=["pre-warmup", "pre-first-average"])
def test_early_vector_checkpoint_resumes_without_loss_reports(tmp_path, memory_warmup):
    kwargs = dict(num_envs=3, memory_size=96, memory_warmup=memory_warmup,
                  batch_size=6, max_train_frames=26, average_loss_freq=100)
    agent = SpaceInvaderAgent(**kwargs)
    try:
        agent.train()
        assert agent.start_frame_num == agent.ReplayMemory.count == 27
        agent.train()
        assert agent.start_frame_num == agent.ReplayMemory.count == 54
        assert agent.frame_nums == agent.averaged_losses == []
        assert bool(agent.losses) is (memory_warmup == 18)
        agent.save(str(tmp_path))
    finally:
        agent.close()

    resumed = SpaceInvaderAgent(**kwargs)
    try:
        resumed.load(str(tmp_path))
        assert resumed.start_frame_num == 54
        assert resumed.losses == agent.losses
        assert resumed.rewards == agent.rewards
        resumed.train()
        assert resumed.start_frame_num == resumed.ReplayMemory.count == 81
        assert resumed.frame_nums == []
    finally:
        resumed.close()


def test_training_writes_metrics_csvs_incrementally(tmp_path):
    metrics_dir = tmp_path / "metrics"
    agent = SpaceInvaderAgent(**AGENT_KWARGS, metrics_dir=str(metrics_dir))
    agent.train()

    episodes_csv = metrics_dir / "episodes.csv"
    losses_csv = metrics_dir / "losses.csv"
    assert episodes_csv.exists()
    assert losses_csv.exists()

    episode_rows = episodes_csv.read_text().splitlines()
    assert episode_rows[0] == "frame_num,episode_num,episode_reward,epsilon,wall_clock_elapsed_seconds"
    assert len(episode_rows) > 1

    loss_rows = losses_csv.read_text().splitlines()
    assert loss_rows[0] == "frame_num,avg_loss"
    assert len(loss_rows) > 1


def test_train_writes_periodic_checkpoint_resumable(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    agent = SpaceInvaderAgent(**AGENT_KWARGS, checkpoint_freq=250, checkpoint_path=str(checkpoint_path))
    agent.train()

    # proves checkpointing fired on its own mid-run, without an explicit agent.save() call
    history_dir = checkpoint_path / "history"
    mid_run_files = list(history_dir.glob("train_history_*"))
    assert len(mid_run_files) == 2
    mid_run_frame = max(int(path.name.rsplit("_", 1)[-1]) for path in mid_run_files)
    assert 0 < mid_run_frame < agent.max_train_frames

    # mirrors train.py's final save(), reproducing the multi-file case load_train_history handles
    agent.save(str(checkpoint_path))
    assert len(list(history_dir.glob("train_history_*"))) > 1

    resumed = SpaceInvaderAgent(**AGENT_KWARGS)
    resumed.load(str(checkpoint_path))

    assert resumed.start_frame_num == agent.start_frame_num == 600
    assert resumed.start_frame_num > mid_run_frame

    resumed.train()
    assert len(resumed.rewards) > 0


def test_train_gates_updates_on_replay_memory_content_not_frame_num():
    """
    A resumed run with an empty replay memory (e.g. after a corrupt/missing snapshot) but a
    frame counter already past memory_warmup must not call update_step() before enough real
    frames exist - doing so hangs forever, since get_batch() can never find a valid index in
    a near-empty buffer.
    """
    agent = SpaceInvaderAgent(**AGENT_KWARGS)
    agent.start_frame_num = AGENT_KWARGS["memory_warmup"] * 10
    agent.max_train_frames = 50

    agent.train()  # hangs before the fix; must return quickly instead

    assert agent.ReplayMemory.count == 50
    assert agent.averaged_losses == []


@pytest.mark.parametrize("double_dqn", [False, True])
def test_agent_resumes_from_saved_checkpoint(tmp_path, double_dqn):
    agent = SpaceInvaderAgent(**AGENT_KWARGS, double_dqn=double_dqn)
    try:
        agent.train()

        losses_before = len(agent.averaged_losses)
        rewards_before = len(agent.rewards)
        last_frame_before = agent.frame_nums[-1]

        agent.save(str(tmp_path))

        resumed = SpaceInvaderAgent(**AGENT_KWARGS, double_dqn=double_dqn)
        try:
            resumed.load(str(tmp_path))

            assert resumed.double_dqn is double_dqn
            # num_envs=1 (the default) means a single sub-buffer, at index 0.
            assert np.array_equal(resumed.ReplayMemory.buffers[0].frames, agent.ReplayMemory.buffers[0].frames)
            assert resumed.ReplayMemory.buffers[0].idx == agent.ReplayMemory.buffers[0].idx
            assert resumed.ReplayMemory.count == agent.ReplayMemory.count
            assert resumed.start_frame_num == last_frame_before

            resumed.train()

            assert len(resumed.averaged_losses) > losses_before
            assert np.isfinite(resumed.averaged_losses).all()
            assert resumed.frame_nums[-1] > last_frame_before
            assert len(resumed.rewards) > rewards_before
        finally:
            resumed.close()
    finally:
        agent.close()
