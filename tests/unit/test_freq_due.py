"""
Unit tests for SpaceInvaderAgent._freq_due (issue #29), which replaces exact-modulo
frequency gating (`frame_num % freq == 0`) now that frame_num advances by num_envs per
tick instead of 1.
"""
from src.agent.agent import SpaceInvaderAgent


def test_matches_original_modulo_check_at_num_envs_one():
    freq = 10
    for frame_num in range(1, 41):
        assert SpaceInvaderAgent._freq_due(frame_num, 1, freq) == (frame_num % freq == 0)


def test_fires_once_when_a_tick_lands_exactly_on_a_multiple():
    # frame_num=100, num_envs=4 covers frames [100, 103]; 100 is itself a multiple of 10.
    assert SpaceInvaderAgent._freq_due(100, 4, 10) is True


def test_fires_once_when_a_tick_crosses_a_multiple_partway_through():
    # frame_num=98, num_envs=4 covers frames [98, 101]; crosses the multiple 100.
    assert SpaceInvaderAgent._freq_due(98, 4, 10) is True


def test_does_not_fire_when_no_multiple_is_covered_by_the_tick():
    # frame_num=101, num_envs=4 covers frames [101, 104]; no multiple of 10 in range.
    assert SpaceInvaderAgent._freq_due(101, 4, 10) is False


def test_fires_at_most_once_even_when_freq_smaller_than_num_envs():
    # freq=4 < num_envs=8: frames [9, 16] cross multiple thresholds (12, 16), still one fire.
    assert SpaceInvaderAgent._freq_due(9, 8, 4) is True
