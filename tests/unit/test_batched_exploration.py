"""
Unit tests for ExplorationVsExploitation.__call__ batched over N states (issue #28).
"""
import numpy as np
import torch

from src.agent.agent import ExplorationVsExploitation
from src.models.cnn import CNNModelPY


def _make_model():
    return CNNModelPY(n_actions=6, device=torch.device("cpu"))


def _make_batch(n, state_shape=(4, 84, 84)):
    return torch.randint(0, 256, (n, *state_shape), dtype=torch.uint8)


def test_single_state_returns_int_unchanged():
    model = _make_model()
    schedule = ExplorationVsExploitation(model, n_actions=6, eps_initial=0.0, eps_final=0.0)
    state = torch.randint(0, 256, (4, 84, 84), dtype=torch.uint8)

    action = schedule(state)

    assert isinstance(action, int)
    assert 0 <= action < 6


def test_batched_epsilon_zero_is_all_greedy_with_one_forward_pass():
    model = _make_model()
    call_count = 0
    original_best_action = model.best_action

    def counting_best_action(x):
        nonlocal call_count
        call_count += 1
        return original_best_action(x)

    model.best_action = counting_best_action

    schedule = ExplorationVsExploitation(model, n_actions=6, eps_initial=0.0, eps_final=0.0)
    batch = _make_batch(8)

    actions = schedule(batch)

    assert len(actions) == 8
    assert all(0 <= a < 6 for a in actions)
    assert call_count == 1


def test_batched_epsilon_one_is_all_random_no_forward_pass():
    model = _make_model()
    call_count = 0
    original_best_action = model.best_action

    def counting_best_action(x):
        nonlocal call_count
        call_count += 1
        return original_best_action(x)

    model.best_action = counting_best_action

    schedule = ExplorationVsExploitation(model, n_actions=6, eps_initial=1.0, eps_final=1.0)
    batch = _make_batch(8)

    actions = schedule(batch)

    assert len(actions) == 8
    assert all(0 <= a < 6 for a in actions)
    assert call_count == 0


def test_batched_mixed_epsilon_splits_explore_and_exploit_per_env(monkeypatch):
    """Monkeypatches shared numpy.random module attributes directly, so this test is not safe to run under parallel pytest execution (e.g. pytest-xdist)."""
    model = _make_model()
    batch = _make_batch(6)
    exploit_idx = [1, 3, 5]
    # Reference greedy actions computed independently of __call__'s batching path.
    expected_exploit_actions = model.best_action(
        batch[exploit_idx].float().div(255.0)
    ).numpy()

    call_count = 0
    exploit_rows_seen = None
    original_best_action = model.best_action

    def counting_best_action(x):
        nonlocal call_count, exploit_rows_seen
        call_count += 1
        exploit_rows_seen = x.shape[0]
        return original_best_action(x)

    model.best_action = counting_best_action

    schedule = ExplorationVsExploitation(model, n_actions=6, eps_initial=0.5, eps_final=0.5)

    # Deterministic mix: envs 0, 2, 4 explore (draw < eps), envs 1, 3, 5 exploit.
    monkeypatch.setattr(
        "src.agent.agent.random.random", lambda n: np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    )
    random_actions = np.array([2, 0, 5, 0, 1, 0])
    monkeypatch.setattr(
        "src.agent.agent.random.randint", lambda n_actions, size: random_actions.copy()
    )

    actions = schedule(batch)

    assert len(actions) == 6
    assert call_count == 1
    assert exploit_rows_seen == 3
    # Exploring envs keep the drawn random action.
    assert actions[0] == random_actions[0]
    assert actions[2] == random_actions[2]
    assert actions[4] == random_actions[4]
    # Exploiting envs got the model's greedy action, not the random draw.
    assert list(actions[exploit_idx]) == list(expected_exploit_actions)
