"""
Unit tests for CNNModelPY's optimizer selection (adam vs rmsprop).
"""
import pytest
import torch
from torch.optim import Adam, RMSprop

from src.models.cnn import CNNModelPY


def test_default_optimizer_is_adam():
    model = CNNModelPY(n_actions=6, device=torch.device("cpu"))
    assert isinstance(model.optimizer, Adam)


def test_optimizer_rmsprop_uses_deepmind_hyperparameters():
    model = CNNModelPY(n_actions=6, device=torch.device("cpu"), optimizer="rmsprop")
    assert isinstance(model.optimizer, RMSprop)

    params = model.optimizer.defaults
    assert params["alpha"] == 0.95
    assert params["eps"] == 0.01
    assert params["centered"] is True
    assert params["momentum"] == 0


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError):
        CNNModelPY(n_actions=6, device=torch.device("cpu"), optimizer="sgd")
