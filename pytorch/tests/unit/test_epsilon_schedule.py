"""
Unit tests for ExplorationVsExploitation's epsilon schedule (see GitHub issue
discussion: the Nature paper anneals epsilon from environment frame 0, independently
of when gradient updates/memory_warmup kick in).
"""
from src.agent.agent import ExplorationVsExploitation


def _make_schedule(**overrides):
    return ExplorationVsExploitation(dqn_model=None, n_actions=6, **overrides)


def test_default_start_fr_is_zero():
    schedule = _make_schedule()
    assert schedule.start_fr == 0


def test_epsilon_anneals_from_the_first_frame():
    schedule = _make_schedule()
    eps_frame_1 = schedule.get_epsilon(1)
    eps_frame_2 = schedule.get_epsilon(2)

    # Strictly decreasing from the very first frame - no plateau at 1.0.
    assert eps_frame_1 < 1.0
    assert eps_frame_2 < eps_frame_1


def test_epsilon_still_at_initial_before_any_frames():
    schedule = _make_schedule()
    assert schedule.get_epsilon(0) == 1.0


def test_epsilon_reaches_final_value_at_end_fr():
    schedule = _make_schedule(end_fr=1_000_000)
    assert schedule.get_epsilon(1_000_000) == 0.1
    assert schedule.get_epsilon(2_000_000) == 0.1


def test_epsilon_independent_of_memory_warmup():
    # The schedule itself has no notion of memory_warmup - annealing should have
    # already progressed well past 1.0 by frame 50_000 even though that's the
    # default point where SpaceInvaderAgent starts running gradient updates.
    schedule = _make_schedule()
    assert schedule.get_epsilon(50_000) < 0.96
