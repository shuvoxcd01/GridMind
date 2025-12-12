import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariDQN(nn.Module):
    def __init__(
        self, observation_shape: tuple, num_actions: int, channel_first: bool = True
    ):
        super(AtariDQN, self).__init__()
        self.channel_first = channel_first

        if self.channel_first:
            channels, height, width = observation_shape
        else:
            height, width, channels = observation_shape

        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the flattened size of the output after the convolutional layers
        test_input = torch.zeros(1, *observation_shape)  # Batch size 1
        conv_out_size = self._get_conv_output_size(test_input)

        self.fc1 = nn.Linear(conv_out_size, 512)  # works if input is 84x84
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_output_size(self, x):
        """Helper function to compute the size of the flattened output after convolutions."""
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.reshape(1, -1).size(1)

    def _add_batch_dim_if_necessary(self, state):
        if state.ndim == 3:
            state = state.unsqueeze(0)
        elif state.ndim != 4:
            raise ValueError(
                f"Expected state to have 3 or 4 dimensions, but got {state.ndim} dimensions."
            )

        return state

    def forward(self, x):
        x = self._add_batch_dim_if_necessary(x)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2)  # from [1, 210, 160, 3] to [1, 3, 210, 160]

        x = x / 255.0  # normalize pixel values to [0, 1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)
