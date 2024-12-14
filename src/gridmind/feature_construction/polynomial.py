from typing import Optional

import numpy as np


class PolynomialFeatureConstructor:
    def __init__(
        self,
        n: int,
        k: int,
        num_features: Optional[int] = None,
        seed: int = 42,
    ):
        """
        n: Maximum Degree of Each Component (output) in the Polynomial Basis
        k: Dimensionality of the (input) State Space (number of components in the input)
        """
        assert n >= 2 and k >= 1

        self.num_features = num_features if num_features is not None else (n + 1) ** k

        assert self.num_features <= (n + 1) ** k

        rng = np.random.default_rng(seed=seed)

        exponents = rng.integers(0, n + 1, size=(self.num_features, k))

        self._feature_constuctor = lambda s: np.prod(np.power(s, exponents), axis=1)

    def __call__(self, state: np.ndarray, *args, **kwds):
        features = self._feature_constuctor(state)
        return features


if __name__ == "__main__":
    state = np.array([1, 0, -1])
    fc = PolynomialFeatureConstructor(n=3, k=3)
    features_1 = fc(state)
    features_2 = fc(state)

    print(features_1)
    print(features_2)
