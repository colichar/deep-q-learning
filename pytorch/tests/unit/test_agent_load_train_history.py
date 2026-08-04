"""
Unit test for SpaceInvaderAgent.load_train_history's checkpoint-selection logic (see GitHub
issue #4: periodic checkpointing calls save() repeatedly against the same path, and since
save_train_history's filename embeds the frame number instead of overwriting a fixed name,
multiple train_history_<frame_num> files end up on disk - load_train_history must pick the
one with the highest frame number, not just glob's first match).
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

    # glob.glob's result order isn't guaranteed to be sorted; force the non-latest-first order
    # here so the test only passes if load_train_history explicitly selects the highest frame
    # number rather than trusting whatever order glob happened to return.
    monkeypatch.setattr("src.agent.agent.glob.glob", lambda pattern: [paths[200], paths[500]])

    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.load_train_history(str(history_dir))

    assert agent.start_frame_num == 500
    assert agent.frame_nums == [500]
    assert agent.averaged_losses == [0.5]
