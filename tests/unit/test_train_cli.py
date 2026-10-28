import sys
from unittest.mock import MagicMock

import pytest

from scripts import train


@pytest.mark.parametrize("double_dqn", [False, True])
def test_double_dqn_argument(double_dqn, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"] + (["--double-dqn"] if double_dqn else []))

    assert train.parse_args().double_dqn is double_dqn


@pytest.mark.parametrize("double_dqn", [False, True])
def test_main_passes_algorithm_to_agent(double_dqn, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train.py"] + (["--double-dqn"] if double_dqn else []))
    agent_class = MagicMock()
    monkeypatch.setattr(train, "SpaceInvaderAgent", agent_class)

    train.main()

    assert agent_class.call_args.kwargs["double_dqn"] is double_dqn
    agent_class.return_value.train.assert_called_once_with()
    agent_class.return_value.close.assert_called_once_with()
    algorithm = "Double DQN" if double_dqn else "DQN"
    assert f"Using algorithm: {algorithm}\n" in capsys.readouterr().out
