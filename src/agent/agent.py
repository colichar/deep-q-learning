from src.utils.preprocessor import Preprocessor
from src.utils.replay_memory import VectorizedReplayMemory
from src.models.cnn import CNNModelPY
from src.utils.space_invaders import NoopResetEnv, make_space_invaders_env, make_space_invaders_training_env

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from torch import where, no_grad, tensor, stack, save, load, device as torch_device
from torch.cuda import is_available
from torchvision.transforms import Resize
import numpy as np
from numpy import mean, random, array, zeros as np_zeros, sign
import matplotlib.pyplot as plt
from PIL import Image
import csv
import glob
import json
import os
import pickle
import time

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
                 start_transition: int = 0,
                 end_transition: int = 1000000,
                 evaluation: bool = False):
        """
        Initiates an ExplorationVsExploitation object with a CNNModel object and hyperparameters.
        """
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.start_transition = start_transition
        self.end_transition = end_transition
        self.slope = (self.eps_initial - self.eps_final) / (self.start_transition - self.end_transition)
        self.intercept = eps_initial - self.slope * self.start_transition

        self.n_actions = n_actions

        self.dqn_model = dqn_model

        self.evaluation = evaluation

    def get_epsilon(self, transition_num: int = 0) -> float:
        """Returns epsilon at the given collected-transition count."""
        if self.evaluation:
            return 0
        elif transition_num <= self.start_transition:
            return self.eps_initial
        elif self.start_transition < transition_num < self.end_transition:
            return self.slope * transition_num + self.intercept
        else:
            return self.eps_final

    def __call__(self, curr_states, transition_num: int = 0):
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
        eps = self.get_epsilon(transition_num)

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
            max_train_transitions=60_000,  # > memory_warmup so a default run actually logs something
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
            double_dqn=False,
    ):
        # Kept as a standalone single env for evaluate()'s gif-recording eval loop,
        # independent of the vectorized training envs below.
        self.my_env = make_space_invaders_env()

        self.num_envs = num_envs
        self.vec_env = self._make_vec_env(num_envs)

        self.completed_transitions = 0

        self.batch_size = batch_size
        self.max_train_transitions = max_train_transitions
        self.memory_warmup = memory_warmup
        self.memory_size = memory_size
        self.update_main_freq = update_main_freq
        self.update_target_freq = update_target_freq
        self.average_loss_freq = average_loss_freq
        self.log_freq = log_freq
        self.discount = discount
        self.double_dqn = double_dqn
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

        # Used to set the restart boundary between loaded replay and fresh new games
        self._replay_restored_unmarked = False

        self.losses = []

        self.transition_nums = []
        self.averaged_losses = []
        self.rewards = []
        self._logged_reward_idx = 0  # index into self.rewards not yet folded into a log print
        self._logged_loss_idx = 0
        self.cumulative_wall_clock_seconds = 0.0

        self.eval_rewards = []
        self.frames_for_gif = []

    @staticmethod
    def _make_vec_env(num_envs):
        """SyncVectorEnv at num_envs == 1 keeps single-env training behaviorally identical
        (no subprocess indirection); AsyncVectorEnv (subprocess-based) is used for N > 1
        so N sub-envs actually step in parallel."""
        env_fns = [make_space_invaders_training_env for _ in range(num_envs)]
        return SyncVectorEnv(env_fns) if num_envs == 1 else AsyncVectorEnv(env_fns)

    @staticmethod
    def _freq_due(transition_num, num_envs, freq):
        """
        Whether a vector tick collecting transitions from `transition_num` through
        `transition_num + num_envs - 1` crosses a scheduling threshold.
        """
        return (transition_num - 1) // freq != (transition_num + num_envs - 1) // freq

    @staticmethod
    def _thresholds_crossed(before, after, freq):
        """Count frequency thresholds in (before, after]."""
        return after // freq - before // freq

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
            if self.double_dqn:
                next_actions = self.MainModel.best_action(new_states)
                target_pred = self.TargetModel(new_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
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
            if self.completed_transitions == 0:
                self._guard_against_unrelated_run(episode_csv_path)
                self._guard_against_unrelated_run(loss_csv_path)
            self._init_metrics_csv(
                episode_csv_path,
                ["transition_num", "episode_num", "episode_reward", "epsilon", "wall_clock_elapsed_seconds"]
            )
            self._init_metrics_csv(loss_csv_path, ["transition_num", "avg_loss"])

        session_start = time.time()
        episode_num = len(self.rewards)

        transition_num = self.completed_transitions + 1
        budget_end = self.max_train_transitions + self.completed_transitions

        if transition_num <= budget_end:
            # Mark restart boundary of loaded memories
            if self._replay_restored_unmarked:
                self.ReplayMemory.mark_restart_boundaries()
                self._replay_restored_unmarked = False
            curr_states, info = self.Preprocessor.initialize_state_vec(self.vec_env)
            prev_lives = array(info["lives"])
            episode_rewards = np_zeros(self.num_envs)
            reset_pending = np.zeros(self.num_envs, dtype=bool)

            while transition_num <= budget_end:
                if reset_pending.any():
                    reset_obs, reset_info = self.vec_env.reset(options={"reset_mask": reset_pending.copy()})
                    reset_frames = [self.Preprocessor.preprocess_frame(reset_obs[i]) for i in np.flatnonzero(reset_pending)]
                    curr_states[reset_pending] = stack(reset_frames).repeat(1, 4, 1, 1)
                    prev_lives[reset_pending] = reset_info["lives"][reset_pending]

                # take actions, one per sub-env
                curr_actions = self.ExploreVsExploit(curr_states, transition_num)

                new_raw_obs, rewards, terminated, truncated, info = self.Preprocessor.step_with_skip_vec(
                    self.vec_env, curr_actions
                )

                lives = array(info["lives"])
                done = terminated | truncated
                life_lost = lives < prev_lives
                prev_lives = lives

                episode_rewards += rewards

                clipped_rewards = sign(rewards)

                # create new sequences with the new frames
                new_states, new_frames = self.Preprocessor.new_state_vec(new_raw_obs, curr_states)

                # store new frames, one per sub-env
                replay_count_before = self.ReplayMemory.count
                self.ReplayMemory.add_frames(
                    new_frames.numpy(), curr_actions, clipped_rewards,
                    life_lost | terminated, boundaries=life_lost | done,
                )
                completed_before = self.completed_transitions
                self.completed_transitions += self.num_envs

                # Resuming without replay starts a new warmup regardless of progress.
                warmed_up = self.ReplayMemory.count > self.memory_warmup

                # Replay may be empty after resume; only thresholds reached with enough
                # newly collected experience are eligible, not missed earlier thresholds.
                last_ineligible_transition = completed_before + max(0, self.memory_warmup - replay_count_before)
                update_count = self._thresholds_crossed(
                    last_ineligible_transition, self.completed_transitions, self.update_main_freq
                ) if warmed_up else 0
                for _ in range(update_count):
                    loss = self.update_step()
                    self.losses.append(loss.item())

                # perform weights update for target model
                if self._freq_due(transition_num, self.num_envs, self.update_target_freq) and warmed_up:
                    self.TargetModel.set_weights(self.MainModel.get_weights())
                    print("Updating target model...")

                if self._freq_due(transition_num, self.num_envs, self.average_loss_freq) and self.losses:
                    self.transition_nums.append(self.completed_transitions)
                    self.averaged_losses.append(mean(self.losses))
                    self.losses = []

                if self._freq_due(transition_num, self.num_envs, self.log_freq):
                    recent_rewards = self.rewards[self._logged_reward_idx:]
                    self._logged_reward_idx = len(self.rewards)
                    avg_reward = f"{mean(recent_rewards):.2f}" if recent_rewards else "n/a (no episode finished yet)"
                    has_new_average = len(self.averaged_losses) > self._logged_loss_idx
                    avg_loss = f"{self.averaged_losses[-1]:.6f}" if has_new_average else "n/a (no new loss average)"
                    self._logged_loss_idx = len(self.averaged_losses)
                    print(
                        f"Finished {self.completed_transitions} transitions. Loss: {avg_loss}. "
                        f"Avg episode reward: {avg_reward}"
                    )
                    if loss_csv_path and has_new_average:
                        self._append_csv_row(loss_csv_path, [self.completed_transitions, self.averaged_losses[-1]])

                if self.checkpoint_path and self._freq_due(transition_num, self.num_envs, self.checkpoint_freq):
                    print(f"Checkpointing at transition {self.completed_transitions}...")
                    write_replay = self._freq_due(transition_num, self.num_envs, self.replay_checkpoint_freq)
                    # Fold elapsed time in before saving, and reset session_start, so a crash
                    # right after this checkpoint doesn't lose this segment's wall-clock time
                    # from cumulative_wall_clock_seconds on the next resume.
                    now = time.time()
                    self.cumulative_wall_clock_seconds += now - session_start
                    session_start = now
                    self.save(self.checkpoint_path, write_replay_memory=write_replay)

                curr_states = new_states
                transition_num += self.num_envs

                # Complete games independently of the remaining lives reported by ALE.
                finished_envs = np.flatnonzero(done)
                if len(finished_envs) > 0:
                    epsilon = self.ExploreVsExploit.get_epsilon(self.completed_transitions)
                    elapsed = self.cumulative_wall_clock_seconds + (time.time() - session_start)
                    for env_idx in finished_envs:
                        episode_num += 1
                        if episode_csv_path:
                            self._append_csv_row(
                                episode_csv_path,
                                [self.completed_transitions, episode_num, episode_rewards[env_idx], epsilon, elapsed]
                            )
                        self.rewards.append(episode_rewards[env_idx])
                        episode_rewards[env_idx] = 0

                reset_pending = done

        self.cumulative_wall_clock_seconds += time.time() - session_start

    def close(self):
        """Closes the vectorized training env and the standalone eval env, so
        AsyncVectorEnv's subprocess workers (num_envs > 1) don't leak."""
        self.vec_env.close()
        self.my_env.close()

    @staticmethod
    def _guard_against_unrelated_run(path):
        # completed_transitions == 0 means this run wasn't resumed; a metrics CSV with data rows here
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
        x_values = self.transition_nums
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
        Saves checkpoint metadata, replay memory, training history and model weights to disk.

        Parameters:
        - path (str): The path where the data should be saved.
        - write_replay_memory (bool): Whether to save the replay memory buffer, which is the
          expensive part of a checkpoint. Set to False to skip it (see replay_checkpoint_freq).
        """

        os.makedirs(path, exist_ok=True)
        with open(path + '/checkpoint_metadata.json', 'w') as file:
            json.dump({"num_envs": self.num_envs}, file)

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
        self.MainModel.save_model(path)
        save(self.TargetModel.get_weights(), path + '/target.pth')

    def save_train_history(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folder '{path}' created.")

        train_history = {
            "completed_transitions": self.completed_transitions,
            "averaged_losses": self.averaged_losses,
            "transition_nums": self.transition_nums,
            "losses": self.losses,
            "rewards": self.rewards,
            "logged_loss_idx": getattr(self, "_logged_loss_idx", 0),
            "cumulative_wall_clock_seconds": self.cumulative_wall_clock_seconds,
        }

        with open(path + f'/train_history_{self.completed_transitions}', 'wb') as file:
            pickle.dump(train_history, file)

    def load(self,
             path
             ):
        """
        Loads the agents replay memory, training history and model weights to disk.

        Parameters:
        - path (str): The path where the data should be saved.
        """

        metadata_path = path + '/checkpoint_metadata.json'
        try:
            with open(metadata_path) as file:
                saved_num_envs = json.load(file).get("num_envs")
        except FileNotFoundError:
            saved_num_envs = None

        if saved_num_envs is None:
            raise ValueError(
                f"Checkpoint '{path}' has no num_envs metadata. Checkpoints without this metadata "
                "are unsupported; start a new run without --resume-from using a new --save-path."
            )
        if saved_num_envs != self.num_envs:
            raise ValueError(
                f"Checkpoint num_envs ({saved_num_envs}) does not match requested num_envs ({self.num_envs}). "
                f"Resume with --num-envs {saved_num_envs}, or start a new run without --resume-from "
                "using a new --save-path."
            )

        print('Loading training history from disk...')
        self.load_train_history(path + '/history')
        print('Training history loaded.')
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
        # In both paths above: marking is a no-op on empty buffers, and restoring real
        # replay means the stored frames predate this run's (always fresh) environments.
        self._replay_restored_unmarked = True
        print('Loading model weights from disk...')
        self.load_model(path + '/model')
        print('Model weights loaded.')

    def load_replay_memory(self, path):
        self.ReplayMemory.load_replay_memory(path)

    def load_model(self, path):
        if not os.path.isfile(path + '/model.pth'):
            raise ValueError(
                f"Checkpoint '{path}' has no model.pth main-network/optimizer state. "
                "Resume from a checkpoint that includes it, or start a new run "
                "without --resume-from using a new --save-path."
            )
        try:
            target_weights = load(path + '/target.pth', map_location=self.TargetModel.device)
        except FileNotFoundError as error:
            raise ValueError(
                f"Checkpoint '{path}' has no target.pth target-network state. "
                "Checkpoints without target state are unsupported; resume from a checkpoint "
                "that includes it, or start a new run without --resume-from using a new --save-path."
            ) from error
        self.MainModel.load_model(path)
        self.TargetModel.set_weights(target_weights)

    def load_train_history(self, path):
        resume_error = (
            "Exact completed-transition progress and training history are required to resume. "
            "Older checkpoints are unsupported; start a new run without --resume-from "
            "using a new --save-path."
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder '{path}' does not exist. {resume_error}")

        matches = glob.glob(os.path.join(path, 'train_history_*'))
        if not matches:
            raise FileNotFoundError(f"No train_history_* file found in '{path}'. {resume_error}")

        # Filename embeds completed transitions so repeated saves accumulate here.
        latest_match = max(matches, key=lambda p: int(p.rsplit('_', 1)[-1]))

        train_history = {}
        with open(latest_match, 'rb') as file:
            train_history = pickle.load(file)

        required_fields = {"completed_transitions", "losses", "averaged_losses", "transition_nums", "rewards"}
        missing_fields = required_fields - train_history.keys()
        if missing_fields:
            raise ValueError(
                f"Training history '{latest_match}' is missing required fields: "
                f"{', '.join(sorted(missing_fields))}. {resume_error}"
            )
        completed_transitions = train_history["completed_transitions"]
        if type(completed_transitions) is not int or completed_transitions < 0:
            raise ValueError(
                f"Training history '{latest_match}' has invalid completed_transitions "
                f"({completed_transitions!r}); expected a non-negative integer. {resume_error}"
            )

        self.losses = train_history["losses"]
        self.averaged_losses = train_history["averaged_losses"]
        self.transition_nums = train_history["transition_nums"]
        self.rewards = train_history["rewards"]
        self._logged_loss_idx = train_history.get("logged_loss_idx", len(self.averaged_losses))
        # Reset the log-print cursor to the end of the restored history, so the first log line
        # after resuming averages only newly-finished episodes, not the whole pre-resume history.
        self._logged_reward_idx = len(self.rewards)
        # .get(..., 0.0) for backward compatibility with pickles saved before this field existed.
        self.cumulative_wall_clock_seconds = train_history.get("cumulative_wall_clock_seconds", 0.0)

        self.completed_transitions = completed_transitions
