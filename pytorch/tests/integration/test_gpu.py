"""
Integration test for GitHub issue #5: confirm the agent/model device resolution
actually lands on the GPU when one is available, rather than silently falling
back to CPU. Skipped on CPU-only machines/CI since there's nothing to verify there.
"""
import pytest
import torch

from src.agent.agent import SpaceInvaderAgent

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_agent_and_models_resolve_to_cuda_when_available():
    agent = SpaceInvaderAgent(memory_size=200, memory_warmup=50, batch_size=16, max_train_frames=10)

    assert agent.device.type == "cuda"
    assert next(agent.MainModel.parameters()).device.type == "cuda"
    assert next(agent.TargetModel.parameters()).device.type == "cuda"
