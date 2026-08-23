"""
Unit test for SpaceInvaderAgent.load_train_history's checkpoint-selection logic (see GitHub
issue #4: periodic checkpointing saves repeatedly to the same path, leaving multiple
train_history_<frame_num> files, so load_train_history must pick the highest one).
"""
import pickle

import pytest

from src.agent.agent import SpaceInvaderAgent


def _write_history(path, frame_num, cumulative_wall_clock_seconds=None):
    history = {"losses": [], "averaged_losses": [0.1], "frame_nums": [frame_num], "rewards": []}
    if cumulative_wall_clock_seconds is not None:
        history["cumulative_wall_clock_seconds"] = cumulative_wall_clock_seconds
    with open(path, "wb") as file:
        pickle.dump(history, file)


@pytest.fixture
def history_dir(tmp_path):
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    return history_dir


def test_load_train_history_restores_cumulative_wall_clock_seconds(history_dir):
    _write_history(history_dir / "train_history_100", frame_num=100, cumulative_wall_clock_seconds=42.5)

    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.load_train_history(str(history_dir))

    assert agent.cumulative_wall_clock_seconds == 42.5


def test_load_train_history_defaults_cumulative_wall_clock_seconds_for_old_pickles(history_dir):
    # Pickles saved before issue #14 don't have this key at all.
    _write_history(history_dir / "train_history_100", frame_num=100)

    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.load_train_history(str(history_dir))

    assert agent.cumulative_wall_clock_seconds == 0.0


def test_save_train_history_writes_cumulative_wall_clock_seconds(history_dir):
    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.losses = []
    agent.averaged_losses = [0.1]
    agent.frame_nums = [100]
    agent.rewards = []
    agent.cumulative_wall_clock_seconds = 12.3

    agent.save_train_history(str(history_dir))

    with open(history_dir / "train_history_100", "rb") as file:
        saved = pickle.load(file)
    assert saved["cumulative_wall_clock_seconds"] == 12.3


def test_load_train_history_picks_highest_frame_num_regardless_of_glob_order(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    history_dir.mkdir()

    paths = {}
    for frame_num, avg_loss in [(200, 0.9), (500, 0.5)]:
        path = history_dir / f"train_history_{frame_num}"
        with open(path, "wb") as file:
            pickle.dump(
                {"losses": [], "averaged_losses": [avg_loss], "frame_nums": [frame_num], "rewards": []},
                file,
            )
        paths[frame_num] = str(path)

    # glob order isn't guaranteed sorted; force non-latest-first so an unsorted-first-match would fail
    monkeypatch.setattr("src.agent.agent.glob.glob", lambda pattern: [paths[200], paths[500]])

    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.load_train_history(str(history_dir))

    assert agent.start_frame_num == 500
    assert agent.frame_nums == [500]
    assert agent.averaged_losses == [0.5]
