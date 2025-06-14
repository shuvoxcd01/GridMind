from typing import Union
import numpy as np
import torch.nn.functional as F
import torch


class OneHotEncoder:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, observation: Union[int, np.ndarray], *args, **kwds):
        if isinstance(observation, np.ndarray):
            num_dims = observation.ndim
            assert num_dims <= 2, "Observation should have at most 2 dimensions."
            if num_dims == 2:
                try:
                    observation = observation.squeeze(axis=-1)
                except ValueError:
                    raise Exception(
                        "Squeezing the last dimension failed. A 2D observation should have a single feature dimension."
                    )

        with torch.no_grad():
            one_hot = F.one_hot(
                torch.tensor(observation, dtype=torch.int64),
                num_classes=self.num_classes,
            )

        return one_hot.cpu().numpy()
