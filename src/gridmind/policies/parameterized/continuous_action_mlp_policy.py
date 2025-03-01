from gridmind.policies.base_policy import BasePolicy
from torch import nn
import math
import torch
import torch.nn.functional as F


class ContinuousActionMLPPolicy(nn.Module, BasePolicy):
    def __init__(
        self,
        observation_shape: tuple,
        num_actions: int,
        num_hidden_layers: int = 0,
        in_features: int = 16,
        out_features: int = 16,
        use_bias: bool = True,
    ):
        nn.Module.__init__(self)
        BasePolicy.__init__(self)

        num_input_features = math.prod(observation_shape)
        self.num_hidden_layers = num_hidden_layers
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = nn.ModuleList()

        if self.num_hidden_layers <= 0:
            self.linear_out = nn.Linear(
                in_features=num_input_features,
                out_features=num_actions * 2,
                bias=use_bias,
            )

        else:
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=num_input_features,
                        out_features=self.out_features,
                        bias=use_bias,
                    ),
                    nn.ReLU(),
                )
            )

            for _ in range(self.num_hidden_layers - 1):
                self.hidden_layers.append(self._create_hidden_layer(use_bias=use_bias))

            self.linear_out = nn.Linear(
                in_features=self.in_features,
                out_features=num_actions * 2,
                bias=use_bias,
            )

    def _create_hidden_layer(self, use_bias: bool):
        return nn.Sequential(
            nn.Linear(self.in_features, self.out_features, bias=use_bias), nn.ReLU()
        )

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        out = self.linear_out(x)

        return out

    def get_action(self, state):
        means, stds = self.get_statistic(state)

        action = torch.normal(mean=means, std=stds).detach().cpu()

        return action

    def get_statistic(self, state):
        logits = self.forward(state)

        means = logits[0::2]
        stds = torch.exp(logits[1::2])

        return means, stds

    def get_action_probs(self, state, action):
        means, stds = self.get_statistic(state)

        density = (1 / (stds * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(
            -((action - means) ** 2) / (2 * stds**2)
        )

        return density

    def update(self, state, action, value):
        pass
