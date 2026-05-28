import itertools
import numbers
from typing import Union

import numpy as np


class PolynomialFeatureConstructor:
    def __init__(
        self,
        n: int,
    ):
        """
        n: Maximum Degree of Each Component (output) in the Polynomial Basis
        """

        C = np.arange(0, n + 1)
        exponents = lambda k: np.array(list(itertools.product(C, repeat=k)))

        self._feature_constuctor = lambda s: np.prod(
            np.power(s, exponents(s.size)), axis=1
        )

    def __call__(self, state: Union[np.ndarray, numbers.Number], *args, **kwds):
        if isinstance(state, numbers.Number):
            state = np.array(state)

        features = self._feature_constuctor(state)
        return features
