"""
Unit test for SpaceInvaderAgent.load_train_history's checkpoint-selection logic (see GitHub
issue #4: periodic checkpointing saves repeatedly to the same path, leaving multiple
train_history_<transition_num> files, so load_train_history must pick the highest one).
"""
import pickle

import pytest

from src.agent.agent import SpaceInvaderAgent


def _write_history(path, transition_num, cumulative_wall_clock_seconds=None):
    history = {
        "completed_transitions": transition_num,
        "losses": [], "averaged_losses": [0.1], "transition_nums": [transition_num], "rewards": [],
    }
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
    _write_history(history_dir / "train_history_100", transition_num=100, cumulative_wall_clock_seconds=42.5)

    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.load_train_history(str(history_dir))

    assert agent.cumulative_wall_clock_seconds == 42.5


def test_load_train_history_defaults_missing_cumulative_wall_clock_seconds(history_dir):
    _write_history(history_dir / "train_history_100", transition_num=100)

    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.load_train_history(str(history_dir))

    assert agent.cumulative_wall_clock_seconds == 0.0


def test_save_train_history_writes_cumulative_wall_clock_seconds(history_dir):
    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.losses = []
    agent.averaged_losses = [0.1]
    agent.transition_nums = [100]
    agent.rewards = []
    agent.cumulative_wall_clock_seconds = 12.3
    agent.completed_transitions = 103

    agent.save_train_history(str(history_dir))

    with open(history_dir / "train_history_103", "rb") as file:
        saved = pickle.load(file)
    assert saved["cumulative_wall_clock_seconds"] == 12.3
    assert saved["completed_transitions"] == 103


def test_load_train_history_picks_highest_transition_num_regardless_of_glob_order(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    history_dir.mkdir()

    paths = {}
    for transition_num, avg_loss in [(200, 0.9), (500, 0.5)]:
        path = history_dir / f"train_history_{transition_num}"
        with open(path, "wb") as file:
            pickle.dump(
                {"completed_transitions": transition_num, "losses": [], "averaged_losses": [avg_loss],
                 "transition_nums": [transition_num - 1], "rewards": []},
                file,
            )
        paths[transition_num] = str(path)

    # glob order isn't guaranteed sorted; force non-latest-first so an unsorted-first-match would fail
    monkeypatch.setattr("src.agent.agent.glob.glob", lambda pattern: [paths[200], paths[500]])

    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.load_train_history(str(history_dir))

    assert agent.completed_transitions == 500
    assert agent.transition_nums == [499]
    assert agent.averaged_losses == [0.5]
