"""
Unit tests for SpaceInvaderAgent.update_step (see GitHub issue #1: the loss must
compare against Q(s, a_taken), not max_a Q(s, a)).
"""
import types

import pytest
import torch

from src.agent.agent import SpaceInvaderAgent


class FakeReplayMemory:
    def __init__(self, batch):
        self._batch = batch

    def get_batch(self):
        return self._batch


class FakeModel:
    """Controlled Q-values for zero-valued current states and one-valued next states."""

    def __init__(self, predictions, next_predictions=None):
        self.predictions = predictions
        self.next_predictions = predictions if next_predictions is None else next_predictions
        self.forward_calls = []
        self.optimizer = types.SimpleNamespace(zero_grad=lambda: None, step=lambda: None)

    def __call__(self, x):
        is_next = bool(torch.all(x == 1))
        self.forward_calls.append((is_next, torch.is_grad_enabled()))
        predictions = self.next_predictions if is_next else self.predictions
        return predictions * 1

    def custom_huber_loss(self, y_pred, y_true):
        self.loss_targets = y_pred
        return torch.nn.functional.huber_loss(y_pred, y_true)

    def best_reward(self, x):
        return self(x).max(dim=1).values

    def best_action(self, x):
        return self(x).argmax(dim=1)


def _make_agent(main_predictions, target_predictions, curr_actions, rewards, terminal_mask,
                double_dqn=False, next_predictions=None):
    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.device = torch.device("cpu")
    agent.discount = 0.99
    agent.double_dqn = double_dqn
    agent.MainModel = FakeModel(main_predictions, next_predictions)
    agent.TargetModel = FakeModel(target_predictions)

    batch_size = curr_actions.shape[0]
    dummy_states = torch.zeros(batch_size, 4, 84, 84, dtype=torch.uint8)
    next_states = torch.full_like(dummy_states, 255)
    agent.ReplayMemory = FakeReplayMemory((dummy_states, next_states, curr_actions, rewards, terminal_mask))
    return agent


@pytest.mark.parametrize("double_dqn", [False, True])
def test_update_step_uses_taken_action_q_value_not_max(double_dqn):
    # Two actions per sample; the max-Q action (index 1, Q=5.0) differs from the
    # taken action (index 0, Q=1.0). If update_step selected by max like
    # `best_reward` does, it would use 5.0 instead of the taken action's Q-value.
    main_predictions = torch.tensor([[1.0, 5.0], [3.0, 2.0]], requires_grad=True)
    target_predictions = torch.zeros(2, 2)
    curr_actions = torch.tensor([0, 0], dtype=torch.int64)
    rewards = torch.tensor([2.0, -1.0])
    terminal_mask = torch.tensor([True, True])

    agent = _make_agent(main_predictions, target_predictions, curr_actions, rewards, terminal_mask, double_dqn)

    loss = agent.update_step()

    taken_action_q = main_predictions.detach().gather(1, curr_actions.unsqueeze(1)).squeeze(1)
    expected_loss = torch.nn.functional.huber_loss(rewards, taken_action_q)
    assert torch.isclose(loss, expected_loss)

    max_action_q = main_predictions.detach().max(dim=1).values
    wrong_loss = torch.nn.functional.huber_loss(rewards, max_action_q)
    assert not torch.isclose(loss, wrong_loss)


@pytest.mark.parametrize("double_dqn", [False, True])
def test_update_step_selects_taken_action_with_single_sample_batch(double_dqn):
    # Same gather-vs-max check as above, but with a batch size of 1, where a
    # squeeze/unsqueeze shape mistake would be most likely to slip through.
    main_predictions = torch.tensor([[1.0, 5.0]], requires_grad=True)
    target_predictions = torch.zeros(1, 2)
    curr_actions = torch.tensor([1], dtype=torch.int64)
    rewards = torch.tensor([0.5])
    terminal_mask = torch.tensor([True])

    agent = _make_agent(main_predictions, target_predictions, curr_actions, rewards, terminal_mask, double_dqn)

    loss = agent.update_step()

    expected_loss = torch.nn.functional.huber_loss(rewards, torch.tensor([5.0]))
    assert torch.isclose(loss, expected_loss)


@pytest.mark.parametrize("double_dqn", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_bootstrap_targets_and_gradient_isolation(double_dqn, batch_size):
    main_predictions = torch.tensor([[1., 5.], [3., 2.], [1., 5.]][:batch_size], requires_grad=True)
    next_predictions = torch.tensor([[9., 1.], [1., 8.], [9., 1.]][:batch_size], requires_grad=True)
    target_predictions = torch.tensor([[2., 7.], [6., 4.], [2., 7.]][:batch_size], requires_grad=True)
    actions = torch.zeros(batch_size, dtype=torch.int64)
    rewards = torch.tensor([1., -1., 3.][:batch_size])
    terminal_mask = torch.tensor([False, False, True][:batch_size])
    agent = _make_agent(main_predictions, target_predictions, actions, rewards, terminal_mask,
                        double_dqn, next_predictions)

    loss = agent.update_step()

    expected_targets = torch.tensor(([2.98, 2.96, 3.] if double_dqn else [7.93, 4.94, 3.])[:batch_size])
    torch.testing.assert_close(agent.MainModel.loss_targets, expected_targets)
    expected_loss = torch.nn.functional.huber_loss(expected_targets, main_predictions.detach()[:, 0])
    torch.testing.assert_close(loss, expected_loss)
    assert not agent.MainModel.loss_targets.requires_grad
    assert target_predictions.grad is None
    assert next_predictions.grad is None
    assert main_predictions.grad is not None
    assert torch.all(main_predictions.grad[:, 0] != 0)
    assert torch.all(main_predictions.grad[:, 1] == 0)
    assert agent.TargetModel.forward_calls == [(True, False)]
    expected_main_calls = [(True, False), (False, True)] if double_dqn else [(False, True)]
    assert agent.MainModel.forward_calls == expected_main_calls
