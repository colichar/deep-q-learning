"""
Unit tests for SpaceInvaderAgent.update_step (see GitHub issue #1: the loss must
compare against Q(s, a_taken), not max_a Q(s, a)).
"""
import types

import torch

from src.agent.agent import SpaceInvaderAgent


class FakeReplayMemory:
    def __init__(self, batch):
        self._batch = batch

    def get_batch(self):
        return self._batch


class FakeModel:
    """Stands in for CNNModelPY: always returns a fixed set of Q-values regardless
    of input, so a test can control exactly which action is the argmax vs which
    action was actually taken."""

    def __init__(self, predictions):
        self.predictions = predictions
        self.optimizer = types.SimpleNamespace(zero_grad=lambda: None, step=lambda: None)

    def __call__(self, x):
        return self.predictions

    def custom_huber_loss(self, y_pred, y_true):
        return torch.nn.functional.huber_loss(y_pred, y_true)

    def best_reward(self, x):
        return self.predictions.max(dim=1).values


def _make_agent(main_predictions, target_predictions, curr_actions, rewards, terminal_mask):
    agent = SpaceInvaderAgent.__new__(SpaceInvaderAgent)
    agent.device = torch.device("cpu")
    agent.discount = 0.99
    agent.MainModel = FakeModel(main_predictions)
    agent.TargetModel = FakeModel(target_predictions)

    batch_size = curr_actions.shape[0]
    dummy_states = torch.zeros(batch_size, 4, 84, 84, dtype=torch.uint8)
    agent.ReplayMemory = FakeReplayMemory((dummy_states, dummy_states, curr_actions, rewards, terminal_mask))
    return agent


def test_update_step_uses_taken_action_q_value_not_max():
    # Two actions per sample; the max-Q action (index 1, Q=5.0) differs from the
    # taken action (index 0, Q=1.0). If update_step selected by max like
    # `best_reward` does, it would use 5.0 instead of the taken action's Q-value.
    main_predictions = torch.tensor([[1.0, 5.0], [3.0, 2.0]], requires_grad=True)
    target_predictions = torch.zeros(2, 2)
    curr_actions = torch.tensor([0, 0], dtype=torch.int64)
    rewards = torch.tensor([2.0, -1.0])
    terminal_mask = torch.tensor([True, True])

    agent = _make_agent(main_predictions, target_predictions, curr_actions, rewards, terminal_mask)

    loss = agent.update_step()

    taken_action_q = main_predictions.detach().gather(1, curr_actions.unsqueeze(1)).squeeze(1)
    expected_loss = torch.nn.functional.huber_loss(rewards, taken_action_q)
    assert torch.isclose(loss, expected_loss)

    max_action_q = main_predictions.detach().max(dim=1).values
    wrong_loss = torch.nn.functional.huber_loss(rewards, max_action_q)
    assert not torch.isclose(loss, wrong_loss)


def test_update_step_moves_actions_to_device():
    main_predictions = torch.tensor([[1.0, 5.0]], requires_grad=True)
    target_predictions = torch.zeros(1, 2)
    curr_actions = torch.tensor([1], dtype=torch.int64)
    rewards = torch.tensor([0.5])
    terminal_mask = torch.tensor([True])

    agent = _make_agent(main_predictions, target_predictions, curr_actions, rewards, terminal_mask)

    # Should not raise even though curr_actions starts off-device-agnostic; update_step
    # is responsible for moving it to self.device before it's used in gather().
    loss = agent.update_step()

    expected_loss = torch.nn.functional.huber_loss(rewards, torch.tensor([5.0]))
    assert torch.isclose(loss, expected_loss)
