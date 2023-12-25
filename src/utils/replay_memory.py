import tensorflow as tf

class ReplayMemory:
    """Stores the experience (frames) of the agent so they can be replayed for training."""
    def __init__(self, max_samples, batch_size):
        self.max_samples = max_samples
        self.batch_size = batch_size

        # Initialize the counter for the buffers
        self.idx = tf.Variable(0, dtype=tf.int32)
        self.num_of_exp = tf.Variable(0, dtype=tf.int32)

        # Initialize the buffers
        self.buffer_current_state = tf.Variable(tf.zeros((max_samples, 84, 84, 4), dtype=tf.uint8))
        self.buffer_next_state = tf.Variable(tf.zeros((max_samples, 84, 84, 4), dtype=tf.uint8))
        self.buffer_action = tf.Variable(tf.zeros((max_samples,), dtype=tf.int32))
        self.buffer_reward = tf.Variable(tf.zeros((max_samples,), dtype=tf.float32))
        self.buffer_lives = tf.Variable(tf.zeros((max_samples,), dtype=tf.int32))

        # Initialize checkpoint for saving/loading
        self.checkpoint = tf.train.Checkpoint(buffer_current_state=self.buffer_current_state,
                                     buffer_next_state=self.buffer_next_state,
                                     buffer_action=self.buffer_action,
                                     buffer_reward=self.buffer_reward,
                                     buffer_lives=self.buffer_lives,
                                     idx=self.idx)
    
    def add_transition(self, transition):
        """
        Adds new transition to replay memory.
        
        Parameters:
        - transition (tuple): The path where the data should be saved.
        """
        current_state, next_state, action, reward, lives = transition
        
        # Update the counter and get the current index
        self.idx.assign(tf.math.mod(self.idx, self.max_samples))

        # Update the buffers
        self.buffer_current_state[self.idx].assign(current_state)
        self.buffer_next_state[self.idx].assign(next_state)
        self.buffer_action[self.idx].assign(action)
        self.buffer_reward[self.idx].assign(reward)
        self.buffer_lives[self.idx].assign(lives)

        # Increment the counter
        self.idx.assign_add(1)
        self.num_of_exp.assign_add(1)
    
    def _batch_random(self):
        """
        Creates a randomly picked training batch from memory.
        """
        # Fix so empty transitions are not selected
        maxval = tf.cond(
            tf.math.less(self.num_of_exp, self.max_samples),
            lambda: self.num_of_exp,
            lambda: self.max_samples
        )

        # randomly batch
        indices = tf.random.uniform(shape=(self.batch_size,), maxval=maxval, dtype=tf.int32)
        sampled_data = (
            tf.gather(self.buffer_current_state, indices),
            tf.gather(self.buffer_next_state, indices),
            tf.gather(self.buffer_action, indices),
            tf.gather(self.buffer_reward, indices),
            tf.gather(self.buffer_lives, indices),
        )

        # Data will be returned as a tuple and not as tf.data.Dataset.
        # The purpose of tf.data.Dataset is often more apparent when
        # dealing with larger datasets that don't fit entirely into
        # memory, and you want to leverage TensorFlow's efficient data
        # loading and preprocessing capabilities.
        # return tf.data.Dataset.from_tensor_slices(sampled_data).batch(self.batch_size)

        return sampled_data

    def save_replay_memory(self, path):
        """
        Saves the agents replay memory to disk.
    
        Parameters:
        - path (str): The path where the data should be saved.
        """
        self.checkpoint.save(path)

    def load_replay_memory(self, path):
        """
        Loads the agents replay memory from disk.
    
        Parameters:
        - path (str): The path frome where the data should be loaded.
        """
        # Restore buffers from disk
        status = self.checkpoint.restore(tf.train.latest_checkpoint(path))

        # Assign the restored values to the buffers
        #self.buffer_current_state.assign(status.buffer_current_state)
        #self.buffer_next_state.assign(status.buffer_next_state)
        #self.buffer_action.assign(status.buffer_action)
        #self.buffer_reward.assign(status.buffer_reward)
        #self.buffer_lives.assign(status.buffer_lives)
    
    @property
    def element_spec(self):
        return (
            tf.TensorSpec(shape=(84, 84, 4), dtype=tf.uint8),
            tf.TensorSpec(shape=(84, 84, 4), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
