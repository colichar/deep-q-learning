"""
Unit test for SpaceInvaderAgent.load_train_history's checkpoint-selection logic (see GitHub
issue #4: periodic checkpointing saves repeatedly to the same path, leaving multiple
train_history_<frame_num> files, so load_train_history must pick the highest one).
"""
import pickle

from src.agent.agent import SpaceInvaderAgent


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
