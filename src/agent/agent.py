from src.utils.preprocessor import Preprocessor
from src.utils.replay_memory import VectorizedReplayMemory
from src.models.cnn import CNNModelPY

import gymnasium as gym
import ale_py
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from torch import where, no_grad, tensor, device as torch_device
from torch.cuda import is_available
from torchvision.transforms import Resize
import numpy as np
from numpy import mean, random, uint8, array, zeros as np_zeros, sign
import matplotlib.pyplot as plt
from PIL import Image
import csv
import glob
import os
import pickle
import time

gym.register_envs(ale_py)


class NoopResetEnv(gym.Wrapper):
    """
    Performs a randomized number of no-op steps inside reset() itself, so each 
    episode starts from a different point in the game's otherwise-fixed opening sequence.
    Living in reset() means gymnasium's NEXT_STEP auto-reset re-randomizes every episode's
    start during training, not just the first.
    """

    def __init__(self, env, noop_action=0):
        super().__init__(env)
        self.noop_action = noop_action

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        n_groups = random.randint(4, 31)
        for _ in range(n_groups):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


def _make_space_invaders_env():
    # Module-level (not a lambda/closure) so AsyncVectorEnv's subprocess workers can
    # pickle it under any multiprocessing start method, not just fork.
    env = gym.make("ALE/SpaceInvaders-v5", frameskip=1, repeat_action_probability=0.0, render_mode="rgb_array")
    return NoopResetEnv(env)


class ExplorationVsExploitation:
    """
    This class handles the epsilon-greedy strategy which will be used to determine whether
    we will choose an action to explore new possibilites or exploit the accumulated experience.
    """

    def __init__(self,
                 dqn_model: CNNModelPY,
                 n_actions: int,
                 eps_initial: float = 1.0,
                 eps_final: float = 0.1,
                 start_fr: int = 0,
                 end_fr: int = 1000000,
                 evaluation: bool = False):
        """
        Initiates an ExplorationVsExploitation object with a CNNModel object and hyperparameters.
        """
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.start_fr = start_fr
        self.end_fr = end_fr
        self.slope = (self.eps_initial - self.eps_final) / (self.start_fr - self.end_fr)
        self.intercept = eps_initial - self.slope * self.start_fr

        self.n_actions = n_actions

        self.dqn_model = dqn_model

        self.evaluation = evaluation

    def get_epsilon(self, frame_num: int = 0) -> float:
        """Returns the current epsilon value for the given frame number, without taking an action."""
        if self.evaluation:
            return 0
        elif frame_num <= self.start_fr:
            return self.eps_initial
        elif self.start_fr < frame_num < self.end_fr:
            return self.slope * frame_num + self.intercept
        else:
            return self.eps_final

    def __call__(self, curr_states, frame_num: int = 0):
        """
        When the object is called, it will return an action to be performed by the agent.
        Accepts either a single state (C, H, W) or a batch of N states (N, C, H, W); each
        state independently explores or exploits per the epsilon-greedy schedule, with the
        exploiting states sharing a single forward pass through the main network. Returns a
        single int for a single state, or an array of N ints for a batch.
        """
        single_state = curr_states.dim() == 3
        if single_state:
            curr_states = curr_states.unsqueeze(0)

        n = curr_states.shape[0]
        eps = self.get_epsilon(frame_num)

        actions = random.randint(self.n_actions, size=n)
        exploit_idx = np.where(random.random(n) >= eps)[0]

        if len(exploit_idx) > 0:
            # we choose the action yielding the highest reward according to our main model
            model_prediction = self.dqn_model.best_action(
                curr_states[exploit_idx].to(self.dqn_model.device).float().div(255.0)
            )
            actions[exploit_idx] = model_prediction.cpu().numpy()

        return int(actions[0]) if single_state else actions


class SpaceInvaderAgent:
    def __init__(
            self,
            learning_rate=0.00025,
            memory_size=10 ** 6,
            memory_warmup=50_000,
            batch_size=32,
            max_train_frames=60_000,  # > memory_warmup so a default run actually logs something
            update_main_freq=4,
            update_target_freq=10_000,
            log_freq=10_000,
            average_loss_freq=400,  # 20
            discount=0.99,
            metrics_dir=None,
            checkpoint_freq=25_000,
            checkpoint_path=None,
            replay_checkpoint_freq=None,
            optimizer="adam",
            num_envs=1,
    ):
        # Kept as a standalone single env for evaluate()'s gif-recording eval loop,
        # independent of the vectorized training envs below.
        self.my_env = NoopResetEnv(gym.make(
            "ALE/SpaceInvaders-v5", frameskip=1, repeat_action_probability=0.0, render_mode="rgb_array"
        ))

        self.num_envs = num_envs
        self.vec_env = self._make_vec_env(num_envs)

        self.start_frame_num = 0

        self.batch_size = batch_size
        self.max_train_frames = max_train_frames
        self.memory_warmup = memory_warmup
        self.memory_size = memory_size
        self.update_main_freq = update_main_freq
        self.update_target_freq = update_target_freq
        self.average_loss_freq = average_loss_freq
        self.log_freq = log_freq
        self.discount = discount
        self.metrics_dir = metrics_dir
        self.checkpoint_freq = checkpoint_freq
        self.replay_checkpoint_freq = replay_checkpoint_freq if replay_checkpoint_freq is not None else checkpoint_freq

        if self.replay_checkpoint_freq % checkpoint_freq != 0:
            raise ValueError(
                f"replay_checkpoint_freq ({self.replay_checkpoint_freq}) must be a multiple of "
                f"checkpoint_freq ({checkpoint_freq}), since the replay-memory write is only "
                "checked when a model/history checkpoint fires."
            )
        self.checkpoint_path = checkpoint_path

        self.device = torch_device("cuda" if is_available() else "cpu")

        self.MainModel = CNNModelPY(self.my_env.action_space.n, learning_rate, device=self.device, optimizer=optimizer)
        self.TargetModel = CNNModelPY(self.my_env.action_space.n, device=self.device)
        self.TargetModel.set_weights(self.MainModel.get_weights())

        self.Preprocessor = Preprocessor()
        self.ReplayMemory = VectorizedReplayMemory(self.num_envs, self.memory_size, self.batch_size)
        self.ExploreVsExploit = ExplorationVsExploitation(self.MainModel, self.my_env.action_space.n)

        self.losses = []

        self.frame_nums = []
        self.averaged_losses = []
        self.rewards = []
        self._logged_reward_idx = 0  # index into self.rewards not yet folded into a log print
        self.cumulative_wall_clock_seconds = 0.0

        self.eval_rewards = []
        self.frames_for_gif = []

    @staticmethod
    def _make_vec_env(num_envs):
        """SyncVectorEnv at num_envs == 1 keeps single-env training behaviorally identical
        (no subprocess indirection); AsyncVectorEnv (subprocess-based) is used for N > 1
        so N sub-envs actually step in parallel."""
        env_fns = [_make_space_invaders_env for _ in range(num_envs)]
        return SyncVectorEnv(env_fns) if num_envs == 1 else AsyncVectorEnv(env_fns)

    @staticmethod
    def _freq_due(frame_num, num_envs, freq):
        """
        Whether a frequency-gated action (originally `frame_num % freq == 0`, back when
        frame_num advanced by 1 per tick) fires for the tick that advances the frame
        counter from `frame_num` through `frame_num + num_envs - 1`. Gating on an exact
        multiple would silently skip past a threshold whenever freq isn't itself a
        multiple of num_envs, so this fires once per threshold actually crossed within
        the tick instead. Reduces to `frame_num % freq == 0` when num_envs == 1.
        """
        return (frame_num - 1) // freq != (frame_num + num_envs - 1) // freq

    def update_step(self):
        minibatch = self.ReplayMemory.get_batch()

        # Extract individual components from minibatch
        curr_states, new_states, curr_actions, rewards, terminal_mask = minibatch

        # Replay memory lives on host RAM regardless of device stored as uint8 (0-255)
        # Transfer as uint8 and cast/scale to [0, 1] on-device, rather than normalizing on CPU
        # saves host-side work and PCIe bandwidth (issue #23).
        curr_states = curr_states.to(self.device).float().div(255.0)
        new_states = new_states.to(self.device).float().div(255.0)
        curr_actions = curr_actions.to(self.device)
        rewards = rewards.to(self.device)
        terminal_mask = terminal_mask.to(self.device)

        # Calculate Q values for non-terminal transitions
        with no_grad():
            target_pred = self.TargetModel.best_reward(new_states)

        target_q = rewards + self.discount * target_pred

        # For terminal transitions, use rewards directly
        target_q = where(terminal_mask, rewards, target_q)

        predictions = self.MainModel(curr_states)
        selected_q_values = predictions.gather(1, curr_actions.long().unsqueeze(1)).squeeze(1)

        loss = self.MainModel.custom_huber_loss(target_q, selected_q_values)

        self.MainModel.optimizer.zero_grad()
        loss.backward()
        self.MainModel.optimizer.step()

        return loss

    def train(self):
        episode_csv_path = None
        loss_csv_path = None
        if self.metrics_dir:
            os.makedirs(self.metrics_dir, exist_ok=True)
            episode_csv_path = os.path.join(self.metrics_dir, "episodes.csv")
            loss_csv_path = os.path.join(self.metrics_dir, "losses.csv")
            if self.start_frame_num == 0:
                self._guard_against_unrelated_run(episode_csv_path)
                self._guard_against_unrelated_run(loss_csv_path)
            self._init_metrics_csv(
                episode_csv_path,
                ["frame_num", "episode_num", "episode_reward", "epsilon", "wall_clock_elapsed_seconds"]
            )
            self._init_metrics_csv(loss_csv_path, ["frame_num", "avg_loss"])

        session_start = time.time()
        episode_num = len(self.rewards)

        frame_num = self.start_frame_num + 1
        budget_end = self.max_train_frames + self.start_frame_num

        if frame_num <= budget_end:
            curr_states, info = self.Preprocessor.initialize_state_vec(self.vec_env)
            prev_lives = array(info["lives"])
            episode_rewards = np_zeros(self.num_envs)
            # Tracks whether each env's running episode_rewards was just flushed this tick,
            # so the leftover-flush below doesn't double-record an env that happened to finish
            # on the very last tick processed.
            flushed_this_tick = np.zeros(self.num_envs, dtype=bool)
            # Holds the *previous* tick's ~alive mask: used to stop mixing frames from
            # two episodes in the same acting state when an env that ended its episode last
            just_reset = np.zeros(self.num_envs, dtype=bool)

            while frame_num <= budget_end:
                # take actions, one per sub-env
                curr_actions = self.ExploreVsExploit(curr_states, frame_num)

                new_raw_obs, rewards, terminated, truncated, info = self.Preprocessor.step_with_skip_vec(
                    self.vec_env, curr_actions
                )

                lives = array(info["lives"])
                alive = lives != 0
                life_lost = lives < prev_lives
                prev_lives = lives

                episode_rewards += rewards

                clipped_rewards = sign(rewards)

                # create new sequences with the new frames
                new_states, new_frames = self.Preprocessor.new_state_vec(new_raw_obs, curr_states)

                # envs whose episode ended, flush frames from previous episode
                if just_reset.any():
                    reset_idx = np.nonzero(just_reset)[0]
                    new_states[reset_idx] = new_frames[reset_idx].unsqueeze(1).repeat(1, 4, 1, 1)

                # store new frames, one per sub-env
                self.ReplayMemory.add_frames(
                    new_frames.numpy(), curr_actions, clipped_rewards, life_lost | ~alive
                )

                # Gate on actual replay-memory content, not frame_num:
                # This way new memory warmup when old replay memory not available
                warmed_up = self.ReplayMemory.count > self.memory_warmup

                # perform weights update for main model
                if self._freq_due(frame_num, self.num_envs, self.update_main_freq) and warmed_up:
                    loss = self.update_step()
                    self.losses.append(loss.item())

                # perform weights update for target model
                if self._freq_due(frame_num, self.num_envs, self.update_target_freq) and warmed_up:
                    self.TargetModel.set_weights(self.MainModel.get_weights())
                    print("Updating target model...")

                # averaging past losses
                if self._freq_due(frame_num, self.num_envs, self.average_loss_freq) and warmed_up:
                    self.frame_nums.append(frame_num)
                    self.averaged_losses.append(mean(self.losses))

                    self.losses = []

                    if self._freq_due(frame_num, self.num_envs, self.log_freq):
                        recent_rewards = self.rewards[self._logged_reward_idx:]
                        self._logged_reward_idx = len(self.rewards)
                        avg_reward = f"{mean(recent_rewards):.2f}" if recent_rewards else "n/a (no episode finished yet)"
                        print(
                            f"Finished {frame_num} frames. Loss: {self.averaged_losses[-1]:.6f}. "
                            f"Avg episode reward: {avg_reward}"
                        )
                        if loss_csv_path:
                            self._append_csv_row(loss_csv_path, [frame_num, self.averaged_losses[-1]])

                if self.checkpoint_path and self._freq_due(frame_num, self.num_envs, self.checkpoint_freq):
                    print(f"Checkpointing at frame {frame_num}...")
                    write_replay = self._freq_due(frame_num, self.num_envs, self.replay_checkpoint_freq)
                    # Fold elapsed time in before saving, and reset session_start, so a crash
                    # right after this checkpoint doesn't lose this segment's wall-clock time
                    # from cumulative_wall_clock_seconds on the next resume.
                    now = time.time()
                    self.cumulative_wall_clock_seconds += now - session_start
                    session_start = now
                    self.save(self.checkpoint_path, write_replay_memory=write_replay)

                curr_states = new_states
                frame_num += self.num_envs

                # a sub-env's episode ends the tick its lives hit 0; log and reset its
                # accumulator, independently of every other sub-env's episode boundary
                flushed_this_tick[:] = False
                finished_envs = np.nonzero(~alive)[0]
                if len(finished_envs) > 0:
                    epsilon = self.ExploreVsExploit.get_epsilon(frame_num)
                    elapsed = self.cumulative_wall_clock_seconds + (time.time() - session_start)
                    for env_idx in finished_envs:
                        episode_num += 1
                        if episode_csv_path:
                            self._append_csv_row(
                                episode_csv_path,
                                [frame_num, episode_num, episode_rewards[env_idx], epsilon, elapsed]
                            )
                        self.rewards.append(episode_rewards[env_idx])
                        episode_rewards[env_idx] = 0
                        flushed_this_tick[env_idx] = True

                just_reset = ~alive

            # Matches the pre-vectorization single-env loop, which always recorded
            # episode_reward once the frame budget ran out, even mid-episode: flush any
            # env's still-in-progress episode now, skipping envs already flushed above on
            # this final tick to avoid double-recording them.
            leftover_envs = np.nonzero(~flushed_this_tick)[0]
            if len(leftover_envs) > 0:
                epsilon = self.ExploreVsExploit.get_epsilon(frame_num)
                elapsed = self.cumulative_wall_clock_seconds + (time.time() - session_start)
                for env_idx in leftover_envs:
                    episode_num += 1
                    if episode_csv_path:
                        self._append_csv_row(
                            episode_csv_path,
                            [frame_num, episode_num, episode_rewards[env_idx], epsilon, elapsed]
                        )
                    self.rewards.append(episode_rewards[env_idx])

        self.cumulative_wall_clock_seconds += time.time() - session_start

    def close(self):
        """Closes the vectorized training env and the standalone eval env, so
        AsyncVectorEnv's subprocess workers (num_envs > 1) don't leak."""
        self.vec_env.close()
        self.my_env.close()

    @staticmethod
    def _guard_against_unrelated_run(path):
        # start_frame_num == 0 means this run wasn't resumed; a metrics CSV with data rows here
        # is from an unrelated prior run, not this one.
        if not os.path.exists(path):
            return
        with open(path, newline="") as file:
            reader = csv.reader(file)
            next(reader, None)
            has_data_row = next(reader, None) is not None
        if has_data_row:
            raise FileExistsError(
                f"'{path}' already has rows from a previous run, but this run wasn't resumed "
                "(no --resume-from). Pass --resume-from to continue that run, or point "
                "--metrics-dir elsewhere to start a fresh one."
            )

    @staticmethod
    def _init_metrics_csv(path, header):
        if not os.path.exists(path):
            with open(path, "w", newline="") as file:
                csv.writer(file).writerow(header)

    @staticmethod
    def _append_csv_row(path, row):
        with open(path, "a", newline="") as file:
            csv.writer(file).writerow(row)

    def evaluate(self, eval_episodes):
        self.eval_rewards = []
        self.frames_for_gif = []
        num_of_ep = 1

        was_evaluation = self.ExploreVsExploit.evaluation
        self.ExploreVsExploit.evaluation = True

        try:
            with no_grad():
                ## start outer loop of the number of episode we'll train the model for
                for episode in range(eval_episodes):

                    episode_reward = 0
                    alive = True

                    ## initialise first sequence of new episode
                    curr_state, _ = self.Preprocessor.initialize_state(self.my_env)

                    while alive:
                        ## choose an exploration/explotation action
                        curr_action = self.ExploreVsExploit(curr_state)

                        ## take action
                        new_raw_obs, reward, terminated, truncated, info = self.Preprocessor.step_with_skip(
                            self.my_env, curr_action
                        )

                        alive = info["lives"] != 0

                        self.frames_for_gif.append(new_raw_obs)

                        episode_reward += reward

                        ## create new sequence with new frame
                        curr_state, _ = self.Preprocessor.new_state(new_raw_obs, curr_state)

                    self.eval_rewards.append(episode_reward)
                    self.export_as_gif(self.frames_for_gif, "eval_" + str(num_of_ep) + "_" + str(episode_reward) + ".gif")
                    self.frames_for_gif = []
                    num_of_ep += 1
        finally:
            self.ExploreVsExploit.evaluation = was_evaluation

        return self.eval_rewards

    def export_as_gif(self, frames: list, name: str):
        resized_frames = [self.resize_frame(frame) for frame in frames]

        images = [Image.fromarray(frame) for frame in resized_frames]

        images[0].save(name, save_all=True, append_images=images[1:], duration=100, loop=0)

    def resize_frame(self, frame):
        # Ensure frame has the correct shape (height, width, channels)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convert frame to uint8 if not already
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            # Resize frame to (420, 320, 3)
            image = Image.fromarray(frame)
            resize_transform = Resize((420, 320))
            resized_image = resize_transform(image)
            # Convert resized image back to numpy array
            resized_frame = np.array(resized_image)
            return resized_frame
        else:
            # Handle unexpected frame shape
            raise ValueError("Invalid frame shape or data type.")

    def plot_history(self):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
        ax1.set_title('Training losses')
        x_values = self.frame_nums
        y_values = self.averaged_losses
        ax1.plot(x_values, y_values)
        ax2.set_title('Training rewards')
        ax2.plot(self.rewards, 'o')
        fig.tight_layout()

    def save(self,
             path,
             write_replay_memory=True,
             ):
        """
        Saves the agents replay memory, training history and model weights to disk.

        Parameters:
        - path (str): The path where the data should be saved.
        - write_replay_memory (bool): Whether to save the replay memory buffer, which is the
          expensive part of a checkpoint. Set to False to skip it (see replay_checkpoint_freq).
        """

        if write_replay_memory:
            print('Saving replay memory to disk...')
            self.save_replay_memory(path + '/replay_memory')
            print('Replay memory saved.')
        print('Saving model to disk...')
        self.save_model(path + '/model')
        print('Model saved.')
        print('Saving training history to disk...')
        self.save_train_history(path + '/history')
        print('Training history saved.')

    def save_replay_memory(self, path):
        # Save replay memory
        self.ReplayMemory.save_replay_memory(path)

    def save_model(self, path):
        # Save model weights
        self.MainModel.save_model(path)

    def save_train_history(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folder '{path}' created.")

        if not self.frame_nums:
            print("No training history to save (no average_loss_freq checkpoint was reached); skipping.")
            return

        train_history = {
            "averaged_losses": self.averaged_losses,
            "frame_nums": self.frame_nums,
            "losses": self.losses,
            "rewards": self.rewards,
            "cumulative_wall_clock_seconds": self.cumulative_wall_clock_seconds,
        }

        with open(path + f'/train_history_{self.frame_nums[-1]}', 'wb') as file:
            pickle.dump(train_history, file)

    def load(self,
             path
             ):
        """
        Loads the agents replay memory, training history and model weights to disk.

        Parameters:
        - path (str): The path where the data should be saved.
        """

        print('Loading replay memory from disk...')
        try:
            self.load_replay_memory(path + '/replay_memory')
            print('Replay memory loaded.')
        except FileNotFoundError:
            print(
                f"No replay-memory snapshot found under '{path}/replay_memory' (possible with "
                "replay_checkpoint_freq > checkpoint_freq if this checkpoint was written before "
                "the first replay-memory save). Continuing with a freshly initialized, empty replay memory."
            )
        print('Loading model weights and training history from disk...')
        self.load_model(path + '/model')
        self.load_train_history(path + '/history')
        print('Model weights and training history loaded.')

    def load_replay_memory(self, path):
        self.ReplayMemory.load_replay_memory(path)

    def load_model(self, path):
        self.MainModel.load_model(path)
        self.TargetModel.set_weights(self.MainModel.get_weights())

    def load_train_history(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder '{path}' does not exist.")

        matches = glob.glob(os.path.join(path, 'train_history_*'))
        if not matches:
            raise FileNotFoundError(f"No train_history_* file found in '{path}'.")

        # filename embeds the frame number, not a fixed name, so repeated saves accumulate here
        latest_match = max(matches, key=lambda p: int(p.rsplit('_', 1)[-1]))

        train_history = {}
        with open(latest_match, 'rb') as file:
            train_history = pickle.load(file)

        print(train_history.keys())

        self.losses = train_history["losses"]
        self.averaged_losses = train_history["averaged_losses"]
        self.frame_nums = train_history["frame_nums"]
        self.rewards = train_history["rewards"]
        # Reset the log-print cursor to the end of the restored history, so the first log line
        # after resuming averages only newly-finished episodes, not the whole pre-resume history.
        self._logged_reward_idx = len(self.rewards)
        # .get(..., 0.0) for backward compatibility with pickles saved before this field existed.
        self.cumulative_wall_clock_seconds = train_history.get("cumulative_wall_clock_seconds", 0.0)

        self.start_frame_num = train_history["frame_nums"][-1]
