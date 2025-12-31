"""
Min-Max normalization for continuous observations.

This module provides a callable MinMaxNormalizer class for normalizing observations
to a target range (default [0, 1]) based on known lower and upper bounds.
"""

from typing import Tuple, Union

import numpy as np


class MinMaxNormalizer:
    """
    Min-Max normalization that scales observations to a target range.

    Transforms observations from [low, high] to [target_min, target_max] using:
        normalized = (obs - low) / (high - low) * (target_max - target_min) + target_min

    This is a callable class that can be used as a feature constructor for
    environments with known observation bounds (e.g., MountainCar, Pendulum).

    Parameters
    ----------
    low : np.ndarray or float
        Lower bounds of the observation space. Can be a scalar (applied to all dimensions)
        or an array matching observation shape.
    high : np.ndarray or float
        Upper bounds of the observation space. Can be a scalar (applied to all dimensions)
        or an array matching observation shape.
    clip : bool, optional
        If True, clip observations to [low, high] before normalizing.
        Useful for handling observations that may occasionally exceed bounds.
        Default is False.
    target_range : Tuple[float, float], optional
        Target range for normalized values as (min, max).
        Default is (0, 1). Use (-1, 1) for tanh-like outputs.
    epsilon : float, optional
        Small value added to denominator to prevent division by zero when low == high.
        Default is 1e-8.

    Examples
    --------
    >>> # Mountain Car environment bounds
    >>> normalizer = MinMaxNormalizer(
    ...     low=np.array([-1.2, -0.07]),
    ...     high=np.array([0.6, 0.07])
    ... )
    >>> obs = np.array([-0.5, 0.0])
    >>> normalized = normalizer(obs)
    >>> print(normalized)  # [0.38888889, 0.5]

    >>> # Normalize to [-1, 1] range
    >>> normalizer = MinMaxNormalizer(
    ...     low=-1.0, high=1.0, target_range=(-1, 1)
    ... )
    >>> obs = np.array([0.0])
    >>> normalized = normalizer(obs)
    >>> print(normalized)  # [0.0]

    >>> # Batch normalization
    >>> normalizer = MinMaxNormalizer(low=0.0, high=10.0)
    >>> batch_obs = np.array([[0.0], [5.0], [10.0]])
    >>> normalized_batch = normalizer(batch_obs)
    >>> print(normalized_batch)  # [[0.0], [0.5], [1.0]]
    """

    def __init__(
        self,
        low: Union[np.ndarray, float],
        high: Union[np.ndarray, float],
        clip: bool = False,
        target_range: Tuple[float, float] = (0.0, 1.0),
        epsilon: float = 1e-8,
    ):
        """
        Initialize the MinMaxNormalizer.

        Parameters
        ----------
        low : np.ndarray or float
            Lower bounds of the observation space.
        high : np.ndarray or float
            Upper bounds of the observation space.
        clip : bool, optional
            Whether to clip observations to [low, high] before normalizing.
        target_range : Tuple[float, float], optional
            Target range for normalized values as (min, max).
        epsilon : float, optional
            Small value to prevent division by zero.

        Raises
        ------
        ValueError
            If any element of low >= high.
        ValueError
            If target_range[0] >= target_range[1].
        """
        # Convert to numpy arrays for consistent handling
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.clip = clip
        self.epsilon = epsilon

        # Validate target_range
        if len(target_range) != 2:
            raise ValueError(
                f"target_range must be a tuple of length 2, got {len(target_range)}"
            )

        self.target_min, self.target_max = target_range

        if self.target_min >= self.target_max:
            raise ValueError(
                f"target_range min ({self.target_min}) must be less than max ({self.target_max})"
            )

        # Validate that low < high
        if np.any(self.low >= self.high):
            raise ValueError(
                f"All elements of low must be strictly less than high. "
                f"Got low={self.low}, high={self.high}"
            )

        # Precompute scaling factors for efficiency
        self.scale = (self.target_max - self.target_min) / (
            self.high - self.low + self.epsilon
        )
        self.offset = self.target_min - self.low * self.scale

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """
        Normalize observation(s) to the target range.

        Parameters
        ----------
        observation : np.ndarray
            Observation(s) to normalize. Can be:
            - 1D array: single observation (shape: (obs_dim,))
            - 2D array: batch of observations (shape: (batch_size, obs_dim))

        Returns
        -------
        np.ndarray
            Normalized observation(s) with the same shape as input.
            Values will be in the range [target_min, target_max].

        Examples
        --------
        >>> normalizer = MinMaxNormalizer(low=0.0, high=10.0)
        >>> # Single observation
        >>> obs = np.array([5.0])
        >>> print(normalizer(obs))  # [0.5]
        >>> # Batch of observations
        >>> batch = np.array([[0.0], [5.0], [10.0]])
        >>> print(normalizer(batch))  # [[0.0], [0.5], [1.0]]
        """
        # Convert to numpy array if not already
        obs = np.asarray(observation, dtype=np.float32)

        # Clip if requested (handles out-of-bounds values)
        if self.clip:
            obs = np.clip(obs, self.low, self.high)

        # Apply normalization: obs * scale + offset
        # Broadcasting handles both single observations and batches
        normalized = obs * self.scale + self.offset

        return normalized

    def __repr__(self) -> str:
        """String representation of the normalizer."""
        return (
            f"MinMaxNormalizer(low={self.low}, high={self.high}, "
            f"clip={self.clip}, target_range=({self.target_min}, {self.target_max}), "
            f"epsilon={self.epsilon})"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("MinMaxNormalizer Examples")
    print("=" * 70)

    # Example 1: Mountain Car environment bounds
    print("\n1. Mountain Car Environment (2D observation)")
    print("-" * 70)
    normalizer = MinMaxNormalizer(
        low=np.array([-1.2, -0.07]), high=np.array([0.6, 0.07])
    )
    print(f"Normalizer: {normalizer}")

    # Single observation
    obs = np.array([-0.5, 0.0])
    normalized = normalizer(obs)
    print(f"\nOriginal observation: {obs}")
    print(f"Normalized to [0, 1]: {normalized}")

    # Batch of observations
    batch_obs = np.array(
        [
            [-1.2, -0.07],  # Min bounds
            [-0.3, 0.0],  # Mid range
            [0.6, 0.07],  # Max bounds
        ]
    )
    normalized_batch = normalizer(batch_obs)
    print(f"\nBatch observations:\n{batch_obs}")
    print(f"Normalized batch:\n{normalized_batch}")

    # Example 2: Clipping out-of-bounds values
    print("\n2. Clipping Out-of-Bounds Values")
    print("-" * 70)
    normalizer_clip = MinMaxNormalizer(
        low=np.array([-1.2, -0.07]), high=np.array([0.6, 0.07]), clip=True
    )

    # Observation with values outside bounds
    obs_oob = np.array([-1.5, 0.1])  # Both dimensions out of bounds
    normalized_clipped = normalizer_clip(obs_oob)
    print(f"Out-of-bounds observation: {obs_oob}")
    print(f"Normalized with clipping: {normalized_clipped}")
    print("(Values clipped to bounds before normalization)")

    # Without clipping (for comparison)
    try:
        normalizer_no_clip = MinMaxNormalizer(
            low=np.array([-1.2, -0.07]), high=np.array([0.6, 0.07]), clip=False
        )
        normalized_no_clip = normalizer_no_clip(obs_oob)
        print(f"Normalized without clipping: {normalized_no_clip}")
        print("(Values can exceed [0, 1] range)")
    except Exception as e:
        print(f"Error without clipping: {e}")

    # Example 3: Custom target range [-1, 1]
    print("\n3. Custom Target Range [-1, 1] (tanh-like)")
    print("-" * 70)
    normalizer_tanh = MinMaxNormalizer(
        low=np.array([-1.2, -0.07]),
        high=np.array([0.6, 0.07]),
        target_range=(-1.0, 1.0),
    )
    print(f"Normalizer: {normalizer_tanh}")

    obs = np.array([-0.3, 0.0])
    normalized_tanh = normalizer_tanh(obs)
    print(f"\nOriginal observation: {obs}")
    print(f"Normalized to [-1, 1]: {normalized_tanh}")

    # Example 4: Scalar bounds (same bounds for all dimensions)
    print("\n4. Scalar Bounds (applied to all dimensions)")
    print("-" * 70)
    normalizer_scalar = MinMaxNormalizer(low=0.0, high=10.0)
    print(f"Normalizer: {normalizer_scalar}")

    obs_3d = np.array([0.0, 5.0, 10.0])
    normalized_3d = normalizer_scalar(obs_3d)
    print(f"\nOriginal observation: {obs_3d}")
    print(f"Normalized: {normalized_3d}")

    # Batch with scalar bounds
    batch_3d = np.array([[0.0, 2.5, 5.0], [5.0, 7.5, 10.0]])
    normalized_batch_3d = normalizer_scalar(batch_3d)
    print(f"\nBatch observations:\n{batch_3d}")
    print(f"Normalized batch:\n{normalized_batch_3d}")

    # Example 5: Edge case - handling very small ranges (epsilon protection)
    print("\n5. Edge Case: Very small range (epsilon protection)")
    print("-" * 70)
    normalizer_edge = MinMaxNormalizer(
        low=1.0,
        high=1.0 + 1e-6,  # Very small but valid range
        epsilon=1e-8,
    )
    obs_edge = np.array([1.0 + 5e-7])  # Value in the middle of tiny range
    normalized_edge = normalizer_edge(obs_edge)
    print(f"Normalizer with very small range: {normalizer_edge}")
    print(f"Original observation: {obs_edge}")
    print(f"Normalized: {normalized_edge}")
    print("(Epsilon helps maintain numerical stability)")

    # Example 6: Demonstration of validation
    print("\n6. Input Validation Examples")
    print("-" * 70)
    try:
        bad_normalizer = MinMaxNormalizer(low=10.0, high=5.0)
    except ValueError as e:
        print("Caught expected error for low >= high:")
        print(f"  {e}")

    try:
        bad_range = MinMaxNormalizer(low=0.0, high=10.0, target_range=(1.0, -1.0))
    except ValueError as e:
        print("\nCaught expected error for invalid target_range:")
        print(f"  {e}")

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
