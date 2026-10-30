"""Checkpoint compatibility tests without ALE environments."""
import json
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.agent.agent import SpaceInvaderAgent
from src.models.cnn import CNNModelPY
from src.utils.replay_memory import VectorizedReplayMemory


def _agent(num_envs, optimizer="adam"):
    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.num_envs = num_envs
    agent.MainModel = CNNModelPY(n_actions=6, device=torch.device("cpu"), optimizer=optimizer)
    agent.TargetModel = CNNModelPY(n_actions=6, device=torch.device("cpu"))
    # Equal sub-buffer capacities reproduce the partial-load bug rather than a shape error.
    agent.ReplayMemory = VectorizedReplayMemory(num_envs, capacity=16 * num_envs, batch_size=4)
    agent.frame_nums = []
    agent.averaged_losses = []
    agent.losses = []
    agent.rewards = []
    agent.cumulative_wall_clock_seconds = 0.0
    agent.start_frame_num = 0
    return agent


def _record_training(agent):
    agent.ReplayMemory.add_frames(
        np.full((agent.num_envs, 84, 84), 42, dtype=np.uint8),
        np.zeros(agent.num_envs, dtype=int),
        np.ones(agent.num_envs),
        np.zeros(agent.num_envs, dtype=bool),
    )
    agent.frame_nums = [100]
    agent.averaged_losses = [0.5]
    agent.rewards = [10.0]


@pytest.mark.parametrize("write_replay_memory", [True, False])
def test_save_records_num_envs_without_training_history(tmp_path, write_replay_memory):
    agent = _agent(2)
    path = tmp_path / "checkpoint"

    agent.save(str(path), write_replay_memory=write_replay_memory)

    assert json.loads((path / "checkpoint_metadata.json").read_text()) == {"num_envs": 2}
    assert not list((path / "history").glob("train_history_*"))
    assert (path / "replay_memory").exists() is write_replay_memory


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("write_replay_memory", [True, False])
@pytest.mark.parametrize("optimizer", ["adam", "rmsprop"])
def test_matching_num_envs_restores_checkpoint(tmp_path, num_envs, write_replay_memory, optimizer):
    agent = _agent(num_envs, optimizer)
    _record_training(agent)
    agent.TargetModel.set_weights(agent.MainModel.get_weights())
    loss = agent.MainModel(torch.rand(2, 4, 84, 84)).square().mean()
    loss.backward()
    agent.MainModel.optimizer.step()
    assert agent.MainModel.optimizer.state
    assert any(
        not torch.equal(weights, agent.TargetModel.state_dict()[name])
        for name, weights in agent.MainModel.state_dict().items()
    )
    agent.save(str(tmp_path), write_replay_memory=write_replay_memory)

    resumed = _agent(num_envs, optimizer)
    resumed.load(str(tmp_path))

    assert resumed.start_frame_num == 100
    assert resumed.averaged_losses == agent.averaged_losses
    assert resumed.rewards == agent.rewards
    assert resumed.ReplayMemory.count == (num_envs if write_replay_memory else 0)
    if write_replay_memory:
        for saved, loaded in zip(agent.ReplayMemory.buffers, resumed.ReplayMemory.buffers):
            np.testing.assert_array_equal(loaded.frames, saved.frames)
            assert loaded.idx == saved.idx
    torch.testing.assert_close(resumed.MainModel.state_dict(), agent.MainModel.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(resumed.TargetModel.state_dict(), agent.TargetModel.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(
        resumed.MainModel.optimizer.state_dict(), agent.MainModel.optimizer.state_dict(), rtol=0, atol=0
    )


def test_missing_target_state_is_unsupported(tmp_path):
    agent = _agent(1)
    _record_training(agent)
    agent.save(str(tmp_path))
    (tmp_path / "model" / "target.pth").unlink()
    resumed = _agent(1)
    resumed.MainModel.load_model = Mock()
    resumed.TargetModel.set_weights = Mock()

    with pytest.raises(ValueError, match="no target.pth target-network state.*without --resume-from"):
        resumed.load(str(tmp_path))

    resumed.MainModel.load_model.assert_not_called()
    resumed.TargetModel.set_weights.assert_not_called()


@pytest.mark.parametrize("saved_num_envs, requested_num_envs", [(1, 2), (2, 1)])
@pytest.mark.parametrize("write_replay_memory", [True, False])
def test_mismatched_num_envs_rejected_before_any_state_load(
    tmp_path, saved_num_envs, requested_num_envs, write_replay_memory, monkeypatch
):
    agent = _agent(saved_num_envs)
    _record_training(agent)
    agent.save(str(tmp_path), write_replay_memory=write_replay_memory)
    resumed = _agent(requested_num_envs)
    loaders = []
    for name in ("load_replay_memory", "load_model", "load_train_history"):
        loader = Mock(wraps=getattr(resumed, name))
        monkeypatch.setattr(resumed, name, loader)
        loaders.append(loader)

    with pytest.raises(ValueError, match=rf"Resume with --num-envs {saved_num_envs}") as error:
        resumed.load(str(tmp_path))

    assert f"requested num_envs ({requested_num_envs})" in str(error.value)
    for loader in loaders:
        loader.assert_not_called()
    assert all(buffer.count == 0 for buffer in resumed.ReplayMemory.buffers)


@pytest.mark.parametrize("metadata", [None, {}])
def test_missing_num_envs_metadata_is_unsupported(tmp_path, metadata):
    if metadata is not None:
        (tmp_path / "checkpoint_metadata.json").write_text(json.dumps(metadata))
    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.num_envs = 1

    with pytest.raises(ValueError, match="no num_envs metadata.*without --resume-from"):
        agent.load(str(tmp_path))
