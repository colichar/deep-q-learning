"""Vector-tick scheduling for single-fire events such as checkpoints."""
from src.agent.agent import SpaceInvaderAgent


def test_matches_original_modulo_check_at_num_envs_one():
    freq = 10
    for transition_num in range(1, 41):
        assert SpaceInvaderAgent._freq_due(transition_num, 1, freq) == (transition_num % freq == 0)


def test_fires_once_when_a_tick_lands_exactly_on_a_multiple():
    # The tick collects transitions [100, 103]; 100 is a multiple of 10.
    assert SpaceInvaderAgent._freq_due(100, 4, 10) is True


def test_fires_once_when_a_tick_crosses_a_multiple_partway_through():
    # The tick collects transitions [98, 101]; it crosses the multiple 100.
    assert SpaceInvaderAgent._freq_due(98, 4, 10) is True


def test_does_not_fire_when_no_multiple_is_covered_by_the_tick():
    # The tick collects transitions [101, 104]; no multiple of 10 is in range.
    assert SpaceInvaderAgent._freq_due(101, 4, 10) is False


def test_boolean_schedule_fires_once_even_when_multiple_thresholds_cross():
    assert SpaceInvaderAgent._freq_due(9, 8, 4) is True


def test_optimizer_threshold_count_is_not_limited_to_one_per_tick():
    assert SpaceInvaderAgent._thresholds_crossed(8, 16, 4) == 2
    assert SpaceInvaderAgent._thresholds_crossed(9, 13, 4) == 1
    assert SpaceInvaderAgent._thresholds_crossed(9, 11, 4) == 0
