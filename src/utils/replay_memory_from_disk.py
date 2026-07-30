import tensorflow as tf
import os

class DiskReplayMemory:
    def __init__(self, capacity, batch_size, data_dir):
        self.capacity = capacity
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.idx = tf.Variable(0, dtype=tf.int32)

    def _serialize_transition(self, transition):
        # Serialize your transition (e.g., using tf.io.serialize_tensor)
        # ...

    def _write_to_disk(self, transition):
        serialized_data = self._serialize_transition(transition)
        file_path = os.path.join(self.data_dir, f"{self.idx.numpy() % self.capacity}.tfrecord")

        # Write to disk (append mode if file exists, create otherwise)
        with tf.io.TFRecordWriter(file_path, options="ZLIB") as writer:
            writer.write(serialized_data)

    def add_transition(self, transition):
        self._write_to_disk(transition)
        self.idx.assign_add(1)

    def _parse_tfrecord_fn(self, example_proto):
        # Implement your parsing logic to convert serialized records back to tensors
        # ...

    def create_dataset(self):
        file_pattern = os.path.join(self.data_dir, "*.tfrecord")
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type="ZLIB"),
            cycle_length=4,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=self.capacity).batch(self.batch_size)
        return dataset
