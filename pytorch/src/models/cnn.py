from torch.nn import Module, Conv2d, Linear
from torch.nn.functional import relu, huber_loss
from torch import max, argmax, save, load
from torch.optim import Adam

import os

class CNNModelPY(Module):
    """
    Implementation of a CNN model to be used by the DQN Agent.
    """
    def __init__(self, n_actions:int, learning_rate:float=0.001, state_shape:tuple=(84,84,4)):
        super(CNNModelPY, self).__init__()
        
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.state_shape = state_shape
        
        self.init_model()
        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def custom_huber_loss(self, y_pred, y_true):
        """
        This is a customized huber loss. The model output y_pred needs to be processed with one further step so
        the loss function can be computed.
        """
        # Ensure y_pred has shape (batch_size, n_actions)
        #y_pred = max(y_pred, dim=1, keepdim=True)[0]

        # Calculate Huber loss using the maximum predicted Q-value
        loss = huber_loss(y_pred, y_true)
        return loss
        
    def init_model(self):
        """
        The same architecture as in the DeepMind Paper. [insert reference here]
        """
        self.conv1 = Conv2d(in_channels=self.state_shape[2], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = Linear(in_features=64*7*7, out_features=512)
        self.fc2 = Linear(in_features=512, out_features=self.n_actions)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def best_action(self, input_data) -> int:
        """
        Function to retrieve the best_action given a current state.
        """
        model_prediction = self.forward(input_data)
        best_action = argmax(model_prediction, dim=1)
        return best_action
    
    def best_reward(self, input_data) -> int:
        """
        Function to retrieve the best expected reward given a current state.
        """
        model_prediction = self.forward(input_data)
        best_reward = max(model_prediction, dim=1).values
        return best_reward

    def get_weights(self):
        """
        Get the model weights.
        """
        return self.state_dict()

    def set_weights(self, weights):
        """
        Set the model weights.
        """
        self.load_state_dict(weights)

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folder '{path}' created.")

        # Save the model and optimizer states to disk
        save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path + '/model.pth')

    def load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder '{path}' does not exist.")
            
        # Load the saved model and optimizer states
        checkpoint = load(path + '/model.pth')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])