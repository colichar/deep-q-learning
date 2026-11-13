from unittest.mock import Mock

from scripts.tensorboard_bridge import sync_losses, sync_rewards


def test_csv_metrics_use_collected_transition_counts_in_tensorboard(tmp_path):
    losses = tmp_path / "losses.csv"
    losses.write_text("transition_num,avg_loss\n4,0.5\n8,0.25\n")
    episodes = tmp_path / "episodes.csv"
    episodes.write_text("transition_num,episode_num,episode_reward,epsilon,wall_clock_elapsed_seconds\n"
                        "5,1,10,0.5,1.0\n8,2,20,0.4,2.0\n")
    writer = Mock()

    assert sync_losses(losses, writer, -1) == 8
    assert sync_rewards(episodes, writer, -1, 2) == 8
    writer.add_scalar.assert_any_call("loss/avg_loss", 0.5, 4)
    writer.add_scalar.assert_any_call("loss/avg_loss", 0.25, 8)
    writer.add_scalar.assert_any_call("reward/avg_episode_reward", 10.0, 5)
    writer.add_scalar.assert_any_call("reward/avg_episode_reward", 15.0, 8)
