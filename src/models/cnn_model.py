from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import huber
from tensorflow import reduce_max, argmax

class CNNModel(Sequential):
    """
    Implementation of a CNN model to be used by the DQN Agent.
    """
    def __init__(self, n_actions:int, learning_rate:float=0.001, state_shape:tuple=(84,84,4)):
        super().__init__()
        
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.state_shape = state_shape
        
        self.init_model()

    def custom_huber_loss(self, y_pred, y_true):
        """
        This is a customized huber loss. The model output y_pred needs to be processed with one further step so
        the loss function can be computed.
        """
        # Ensure y_pred has shape (batch_size, n_actions)
        y_pred = reduce_max(y_pred, axis=-1, keepdims=True)

        # Calculate Huber loss using the maximum predicted Q-value
        loss = huber(y_true, y_pred, delta=1.0)

        return loss
        
    def init_model(self):
        """
        The same architecture as in the DeepMind Paper. [insert reference here]
        """
        self.add(Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding="same", activation="relu", input_shape=self.state_shape))
        self.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", activation="relu"))
        self.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
        self.add(Flatten())
        self.add(Dense(512, activation="relu"))
        self.add(Dense(self.n_actions, activation="linear"))

        self.compile(loss=self.custom_huber_loss, optimizer=Adam(learning_rate=self.learning_rate))
        
    def best_action(self, input_data) -> int:
        """
        Function to retrieve the best_action given a current state.
        """
        model_prediction = self.predict(input_data, verbose=0)
        model_prediction = argmax(model_prediction, axis=1)
        return model_prediction
    
    def best_reward(self, input_data) -> int:
        """
        Function to retrieve the best expected reward given a current state.
        """
        model_prediction = self.predict(input_data, verbose=0)
        model_prediction = reduce_max(model_prediction, axis=1)
        return model_prediction