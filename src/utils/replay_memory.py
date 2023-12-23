import tensorflow as tf
#from tensorflow import data.Dataset

class ReplayMemory(tf.data.Dataset):
    def __init__(self, max_samples, batch_size):
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.idx = 0
        
        self.buffer_current_state = tf.TensorArray(tf.uint8, size=max_samples)
        self.buffer_next_state = tf.TensorArray(tf.uint8, size=max_samples)
        self.buffer_action = tf.TensorArray(tf.int32, size=max_samples)
        self.buffer_reward = tf.TensorArray(tf.float32, size=max_samples)
        self.buffer_lives = tf.TensorArray(tf.int32, size=max_samples)
    
    def add_transition(self, transition):
        if (self.idx + 1) == self.max_samples:
            self.idx = 0
        
        current_state, next_state, action, reward, lives = transition
        self.buffer_current_state = self.buffer_current_state.write(self.idx, current_state)
        self.buffer_next_state = self.buffer_next_state.write(self.idx, next_state)
        self.buffer_action = self.buffer_action.write(self.idx, action)
        self.buffer_reward = self.buffer_reward.write(self.idx, reward)
        self.buffer_lives = self.buffer_lives.write(self.idx, lives)
    
        self.idx += 1
    
    
    def _inputs(self):
        return []
    
    @property
    def element_spec(self):
        return (
            tf.TensorSpec(shape=(84, 84, 4), dtype=tf.uint8, name='current_state'),
            tf.TensorSpec(shape=(84, 84, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    
    def _as_variant_tensor(self):
        """
        This method converts the dataset into a variant tensor.
        It returns a variant tensor that represents the state of the dataset.
        The returned tensor is often used internally by TensorFlow to manage
        datasets within the graph.
        """
        return tf.raw_ops.TensorListToTensor(
            element_shape= [
                tf.TensorShape([84, 84, 4]),
                tf.TensorShape([84, 84, 1]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([])
            ],
            tensor_lists=[
                self.buffer_current_state.stack(),
                self.buffer_next_state.stack(),
                tf.cast(self.buffer_action.stack(), tf.float32),  # Casting to float32 for consistency
                self.buffer_reward.stack(),
                tf.cast(self.buffer_action.stack(), tf.float32),  # Casting to float32 for consistency
            ]
        )
    
    def _batch_random(self):
        """
        Creates a randomly picked training batch from memory.
        """
        indices = tf.random.uniform(shape=(self.batch_size,), maxval=self.max_samples, dtype=tf.int32)
        sampled_data = (
            tf.gather(self.buffer_current_state.stack(), indices),
            tf.gather(self.buffer_next_state.stack(), indices),
            tf.gather(self.buffer_action.stack(), indices),
            tf.gather(self.buffer_reward.stack(), indices),
            tf.gather(self.buffer_lives.stack(), indices),
        )

        # Data will be returned as a tuple and not as tf.data.Dataset.
        # The purpose of tf.data.Dataset is often more apparent when
        # dealing with larger datasets that don't fit entirely into
        # memory, and you want to leverage TensorFlow's efficient data
        # loading and preprocessing capabilities.
        # return tf.data.Dataset.from_tensor_slices(sampled_data).batch(self.batch_size)

        return sampled_data
    
    def _as_dataset(self, *args, **kwargs):
        return self
    
    def _variant_tensor_attr(self):
        return self._as_variant_tensor()
    
    def __iter__(self):
        return zip(
            self.buffer_current_state.stack(),
            self.buffer_next_state.stack(),
            self.buffer_action.stack(),
            self.buffer_reward.stack(),
            self.buffer_lives.stack()
        )
    
    def get_buffer(self):
        return (
            self.buffer_current_state.stack(),
            self.buffer_next_state.stack(),
            self.buffer_action.stack(),
            self.buffer_reward.stack(),
            self.buffer_lives.stack()
        )