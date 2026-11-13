"""
Unit tests for epsilon annealing by collected transitions, independent of warmup.
"""
from src.agent.agent import ExplorationVsExploitation


def _make_schedule(**overrides):
    return ExplorationVsExploitation(dqn_model=None, n_actions=6, **overrides)


def test_default_start_transition_is_zero():
    schedule = _make_schedule()
    assert schedule.start_transition == 0


def test_epsilon_anneals_from_the_first_transition():
    schedule = _make_schedule()
    eps_transition_1 = schedule.get_epsilon(1)
    eps_transition_2 = schedule.get_epsilon(2)

    assert eps_transition_1 < 1.0
    assert eps_transition_2 < eps_transition_1


def test_epsilon_still_at_initial_before_any_transitions():
    schedule = _make_schedule()
    assert schedule.get_epsilon(0) == 1.0


def test_epsilon_reaches_final_value_at_end_transition():
    schedule = _make_schedule(end_transition=1_000_000)
    assert schedule.get_epsilon(1_000_000) == 0.1
    assert schedule.get_epsilon(2_000_000) == 0.1


def test_epsilon_independent_of_memory_warmup():
    # The schedule itself has no notion of memory_warmup - annealing should have
    # already progressed well past 1.0 by transition 50_000 even though that's the
    # default point where SpaceInvaderAgent starts running gradient updates.
    schedule = _make_schedule()
    assert schedule.get_epsilon(50_000) < 0.96
