"""
Unit tests for SpaceInvaderAgent._guard_against_unrelated_run (see GitHub issue #13: a fresh,
non-resumed train() run must not silently append onto a metrics CSV left over from a prior,
unrelated run).
"""
import csv

import pytest

from src.agent.agent import SpaceInvaderAgent


def _write_csv(path, header, rows=()):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def test_guard_allows_missing_file(tmp_path):
    SpaceInvaderAgent._guard_against_unrelated_run(str(tmp_path / "episodes.csv"))


def test_guard_allows_header_only_file(tmp_path):
    path = tmp_path / "episodes.csv"
    _write_csv(path, ["frame_num", "episode_num"])

    SpaceInvaderAgent._guard_against_unrelated_run(str(path))


def test_guard_rejects_file_with_data_rows(tmp_path):
    path = tmp_path / "episodes.csv"
    _write_csv(path, ["frame_num", "episode_num"], rows=[[100, 1]])

    with pytest.raises(FileExistsError):
        SpaceInvaderAgent._guard_against_unrelated_run(str(path))


def test_train_raises_before_resume_when_metrics_dir_has_prior_run(tmp_path, monkeypatch):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    _write_csv(metrics_dir / "episodes.csv",
               ["frame_num", "episode_num", "episode_reward", "epsilon", "wall_clock_elapsed_seconds"],
               rows=[[100, 1, 5.0, 0.5, 1.0]])

    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.metrics_dir = str(metrics_dir)
    agent.start_frame_num = 0

    with pytest.raises(FileExistsError):
        agent.train()
