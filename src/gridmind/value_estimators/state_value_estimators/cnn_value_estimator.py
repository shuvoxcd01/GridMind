from gridmind.policies.base_policy import BasePolicy
from sklearn.base import BaseEstimator
from torch import nn
import math
import torch
import torch.nn.functional as F


class CNNValueEstimator(BaseEstimator):
    def __init__(
        self,
        observation_shape: tuple,
    ):
        super(CNNValueEstimator, self).__init__()

        H, W, C = observation_shape

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=C, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Calculate the flattened size of the output after the convolutional layers
        test_input = torch.zeros(1, *observation_shape)  # Batch size 1
        conv_out_size = self._get_conv_output_size(test_input)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, 1)

    def _get_conv_output_size(self, x):
        """Helper function to compute the size of the flattened output after convolutions."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))

        return self.fc2(x)

    def get_action(self, state):
        action_probs = self.forward(state)

        action_probs = F.softmax(action_probs, dim=-1)

        action = torch.multinomial(action_probs, num_samples=1).detach().cpu().item()

        return action

    def get_action_probs(self, state, action):
        action_probs = self.forward(state)

        action_probs = F.softmax(action_probs, dim=-1)

        return action_probs[action]

    def update(self, state, action, value):
        pass
