"""Checkpoint compatibility tests without ALE environments."""
import json
import pickle
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.agent.agent import SpaceInvaderAgent
from src.models.cnn import CNNModelPY
from src.utils.replay_memory import VectorizedReplayMemory


def _agent(num_envs):
    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.num_envs = num_envs
    agent.MainModel = CNNModelPY(n_actions=6, device=torch.device("cpu"))
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
    agent.start_frame_num = 103
    agent.averaged_losses = [0.5]
    agent.rewards = [10.0]


@pytest.mark.parametrize("write_replay_memory", [True, False])
def test_save_before_training_is_resumable(tmp_path, write_replay_memory):
    agent = _agent(2)
    path = tmp_path / "checkpoint"

    agent.save(str(path), write_replay_memory=write_replay_memory)

    assert json.loads((path / "checkpoint_metadata.json").read_text()) == {"num_envs": 2}
    assert (path / "history" / "train_history_0").exists()
    assert (path / "replay_memory").exists() is write_replay_memory
    resumed = _agent(2)
    resumed.load(str(path))
    assert resumed.start_frame_num == 0
    assert resumed.losses == resumed.frame_nums == resumed.averaged_losses == resumed.rewards == []


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("write_replay_memory", [True, False])
def test_matching_num_envs_restores_checkpoint(tmp_path, num_envs, write_replay_memory):
    agent = _agent(num_envs)
    _record_training(agent)
    agent.save(str(tmp_path), write_replay_memory=write_replay_memory)

    resumed = _agent(num_envs)
    resumed.load(str(tmp_path))

    assert resumed.start_frame_num == 103
    assert resumed.averaged_losses == agent.averaged_losses
    assert resumed.rewards == agent.rewards
    assert resumed.ReplayMemory.count == (num_envs if write_replay_memory else 0)
    if write_replay_memory:
        for saved, loaded in zip(agent.ReplayMemory.buffers, resumed.ReplayMemory.buffers):
            np.testing.assert_array_equal(loaded.frames, saved.frames)
            assert loaded.idx == saved.idx
    for name, weights in agent.MainModel.state_dict().items():
        torch.testing.assert_close(resumed.MainModel.state_dict()[name], weights)
        torch.testing.assert_close(resumed.TargetModel.state_dict()[name], weights)


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


def _training_agent(num_envs, memory_warmup, average_loss_freq):
    agent = _agent(num_envs)
    states = torch.zeros(num_envs, 4, 84, 84, dtype=torch.uint8)
    frames = torch.zeros(num_envs, 84, 84, dtype=torch.uint8)
    info = {"lives": np.full(num_envs, 3)}
    agent.vec_env = None
    agent.Preprocessor = Mock()
    agent.Preprocessor.initialize_state_vec.return_value = (states, info)
    agent.Preprocessor.step_with_skip_vec.return_value = (
        None, np.ones(num_envs), None, None, info,
    )
    agent.Preprocessor.new_state_vec.return_value = (states, frames)
    agent.ExploreVsExploit = Mock(return_value=np.zeros(num_envs, dtype=int))
    agent.ExploreVsExploit.get_epsilon.return_value = 0.5
    agent.update_step = Mock(return_value=torch.tensor(0.5))
    agent.metrics_dir = None
    agent.checkpoint_path = None
    agent.checkpoint_freq = 5
    agent.replay_checkpoint_freq = 10
    agent.memory_warmup = memory_warmup
    agent.update_main_freq = 1
    agent.update_target_freq = 100
    agent.average_loss_freq = average_loss_freq
    agent.log_freq = 100
    agent.max_train_frames = 8
    return agent


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("memory_warmup, average_loss_freq", [
    (100, 100), (0, 100), (0, 7),
], ids=["pre-warmup", "pre-first-average", "between-reports"])
def test_exact_progress_through_periodic_save_repeat_and_resume(
    tmp_path, num_envs, memory_warmup, average_loss_freq,
):
    agent = _training_agent(num_envs, memory_warmup, average_loss_freq)
    agent.checkpoint_path = str(tmp_path / "periodic")
    agent.metrics_dir = str(tmp_path / "metrics")

    agent.train()

    collected = 9 if num_envs == 3 else 8
    checkpoint_count = 6 if num_envs == 3 else 5
    assert agent.start_frame_num == agent.ReplayMemory.count == collected
    assert agent.Preprocessor.initialize_state_vec.call_count == 1
    with open(tmp_path / f"periodic/history/train_history_{checkpoint_count}", "rb") as file:
        periodic_history = pickle.load(file)
    assert periodic_history["completed_transitions"] == checkpoint_count
    assert not (tmp_path / "periodic/replay_memory").exists()

    resumed = _training_agent(num_envs, memory_warmup, average_loss_freq)
    resumed.load(agent.checkpoint_path)
    assert resumed.start_frame_num == checkpoint_count
    assert resumed.losses == periodic_history["losses"]
    assert resumed.frame_nums == periodic_history["frame_nums"]
    resumed.train()
    assert resumed.ExploreVsExploit.call_args_list[0].args[1] == checkpoint_count + 1
    assert resumed.start_frame_num == checkpoint_count + collected

    agent.train()
    assert agent.start_frame_num == agent.ReplayMemory.count == 2 * collected
    assert [call.args[1] for call in agent.ExploreVsExploit.call_args_list] == list(
        range(1, 2 * collected + 1, num_envs)
    )
    if memory_warmup:
        assert agent.losses == agent.averaged_losses == agent.frame_nums == []
    elif average_loss_freq == 100:
        assert agent.losses
        assert agent.frame_nums == []
    else:
        assert agent.frame_nums[-1] < agent.start_frame_num

    agent.save(str(tmp_path / "final"))
    resumed.load(str(tmp_path / "final"))
    assert resumed.start_frame_num == 2 * collected
    assert resumed.losses == agent.losses
    assert resumed.frame_nums == agent.frame_nums
    assert resumed.averaged_losses == agent.averaged_losses
    assert resumed.rewards == agent.rewards
    assert resumed.cumulative_wall_clock_seconds == agent.cumulative_wall_clock_seconds


@pytest.mark.parametrize("has_history", [True, False])
def test_legacy_history_rejected_before_loading_models_or_replay(tmp_path, has_history):
    (tmp_path / "checkpoint_metadata.json").write_text('{"num_envs": 1}')
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    if has_history:
        with open(history_dir / "train_history_100", "wb") as file:
            pickle.dump({"losses": [], "averaged_losses": [0.5], "frame_nums": [100], "rewards": []}, file)
    agent = _agent(1)
    agent.load_model = Mock()
    agent.load_replay_memory = Mock()

    error = ValueError if has_history else FileNotFoundError
    with pytest.raises(error, match="Older checkpoints are unsupported.*new --save-path"):
        agent.load(str(tmp_path))

    agent.load_model.assert_not_called()
    agent.load_replay_memory.assert_not_called()
    assert agent.start_frame_num == 0
