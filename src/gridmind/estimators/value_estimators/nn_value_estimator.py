from torch import nn
import math


class NNValueEstimator(nn.Module):
    def __init__(
        self,
        observation_shape: tuple,
        num_hidden_layers: int = 0,
        use_bias: bool = True,
    ):
        super().__init__()
        num_in_features = math.prod(observation_shape)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers = nn.ModuleList()

        if self.num_hidden_layers <= 0:
            self.linear_out = nn.Linear(
                in_features=num_in_features, out_features=1, bias=use_bias
            )

        else:
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=num_in_features, out_features=16, bias=use_bias
                    ),
                    nn.ReLU(),
                )
            )

            for _ in range(self.num_hidden_layers - 1):
                self.hidden_layers.append(self._create_hidden_layer(use_bias=use_bias))

            self.linear_out = nn.Linear(in_features=16, out_features=1, bias=use_bias)

    def _create_hidden_layer(self, use_bias: bool):
        return nn.Sequential(nn.Linear(16, 16, bias=use_bias), nn.ReLU())

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        out = self.linear_out(x)

        return out
