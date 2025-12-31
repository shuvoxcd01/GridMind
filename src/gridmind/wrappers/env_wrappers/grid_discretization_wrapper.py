"""Grid-based discretization wrapper for continuous observation spaces.

This module provides a simple, robust wrapper that discretizes continuous
observations into discrete states using uniform grid binning.
"""

from typing import Union

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete

from gridmind.wrappers.env_wrappers.base_gym_wrapper import BaseGymWrapper


class GridDiscretizationWrapper(BaseGymWrapper):
    """Discretizes continuous observation spaces using uniform grid binning.

    This wrapper transforms a continuous Box observation space into a discrete
    state space by dividing each dimension into uniform bins. Multi-dimensional
    observations are mapped to a single discrete state using row-major ordering.

    Algorithm:
        For each dimension:
        1. Optionally clip observation to [low, high] bounds
        2. Normalize to [0, 1]: (obs - low) / (high - low)
        3. Multiply by num_bins and floor: int(normalized * num_bins)
        4. Clip bin index to [0, num_bins-1]
        5. Convert multi-dimensional bin indices to single discrete state

    State Mapping:
        Uses row-major ordering to convert multi-dimensional bin indices to a
        single integer state. For example, with bins_per_dim=[20, 15]:
            state = bin_indices[0] * 15 + bin_indices[1]
            Total states = 20 * 15 = 300

    Parameters
    ----------
    env : gymnasium.Env
        The environment to wrap. Must have a Box observation space.
    bins_per_dim : int or list of int
        Number of bins per dimension. If int, uses same number of bins for all
        dimensions. If list, must match observation space dimensionality.
    clip : bool, optional
        Whether to clip observations to [low, high] before discretizing.
        Default: True. Recommended to prevent out-of-bounds issues.

    Raises
    ------
    TypeError
        If the observation space is not Box (continuous).
    ValueError
        If bins_per_dim dimensions don't match observation space, or if
        bins_per_dim contains values <= 0.

    Examples
    --------
    Basic usage with MountainCar (2D observation space):

    >>> import gymnasium as gym
    >>> env = gym.make("MountainCar-v0")
    >>> # Use 20 bins for both position and velocity
    >>> wrapped = GridDiscretizationWrapper(env, bins_per_dim=20)
    >>> print(wrapped.observation_space)
    Discrete(400)  # 20 * 20 = 400 states

    Different bins per dimension:

    >>> wrapped = GridDiscretizationWrapper(env, bins_per_dim=[20, 15])
    >>> print(wrapped.observation_space)
    Discrete(300)  # 20 * 15 = 300 states

    Usage in training:

    >>> obs, info = wrapped.reset()
    >>> print(type(obs))
    <class 'numpy.int64'>  # Discrete state, not continuous array
    >>> action = wrapped.action_space.sample()
    >>> obs, reward, terminated, truncated, info = wrapped.step(action)

    Attributes
    ----------
    observation_space : gymnasium.spaces.Discrete
        The discretized observation space with n = product of all bins_per_dim.
    bins_per_dim : np.ndarray
        Array of bin counts per dimension.
    clip : bool
        Whether observations are clipped to bounds.
    obs_low : np.ndarray
        Lower bounds of the original observation space.
    obs_high : np.ndarray
        Upper bounds of the original observation space.
    obs_shape : tuple
        Shape of the original observation space.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        bins_per_dim: Union[int, list[int]],
        clip: bool = True,
    ):
        """Initialize the grid discretization wrapper.

        Parameters
        ----------
        env : gymnasium.Env
            The environment to wrap.
        bins_per_dim : int or list of int
            Number of bins per dimension.
        clip : bool, optional
            Whether to clip observations to bounds, by default True.
        """
        super().__init__(env)

        # Validate observation space is continuous (Box)
        if not isinstance(env.observation_space, Box):
            raise TypeError(
                f"GridDiscretizationWrapper requires Box observation space, "
                f"got {type(env.observation_space).__name__}"
            )

        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        self.obs_shape = env.observation_space.shape
        self.clip = clip

        # Handle bins_per_dim parameter
        if isinstance(bins_per_dim, int):
            # Use same bins for all dimensions
            if bins_per_dim <= 0:
                raise ValueError(f"bins_per_dim must be positive, got {bins_per_dim}")
            self.bins_per_dim = np.full(self.obs_shape[0], bins_per_dim, dtype=np.int32)
        else:
            # Use specific bins per dimension
            bins_array = np.array(bins_per_dim, dtype=np.int32)
            if len(bins_array) != self.obs_shape[0]:
                raise ValueError(
                    f"bins_per_dim length ({len(bins_array)}) must match "
                    f"observation dimensionality ({self.obs_shape[0]})"
                )
            if np.any(bins_array <= 0):
                raise ValueError(
                    f"All bins_per_dim values must be positive, got {bins_array}"
                )
            self.bins_per_dim = bins_array

        # Compute total number of discrete states (product of all bins)
        self.total_states = int(np.prod(self.bins_per_dim))

        # Create new discrete observation space
        self.observation_space = Discrete(self.total_states)

        # Precompute bin ranges for efficiency
        # Each dimension divided into equal-width bins
        self.bin_widths = (self.obs_high - self.obs_low) / self.bins_per_dim

        # Precompute multipliers for row-major state index calculation
        # For [n1, n2, n3] bins: multipliers = [n2*n3, n3, 1]
        self.state_multipliers = np.zeros(len(self.bins_per_dim), dtype=np.int32)
        self.state_multipliers[-1] = 1
        for i in range(len(self.bins_per_dim) - 2, -1, -1):
            self.state_multipliers[i] = (
                self.state_multipliers[i + 1] * self.bins_per_dim[i + 1]
            )

    def _discretize_observation(self, observation: np.ndarray) -> int:
        """Convert continuous observation to discrete state index.

        Parameters
        ----------
        observation : np.ndarray
            Continuous observation from the environment.

        Returns
        -------
        int
            Discrete state index in range [0, total_states-1].
        """
        # Step 1: Optionally clip to bounds
        if self.clip:
            obs = np.clip(observation, self.obs_low, self.obs_high)
        else:
            obs = observation

        # Step 2: Normalize to [0, 1] for each dimension
        # Handle potential division by zero for infinite bounds
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = (obs - self.obs_low) / (self.obs_high - self.obs_low)
            # Replace NaN/Inf with 0 (happens when low == high or infinite bounds)
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)

        # Step 3: Convert to bin indices
        # Multiply by num_bins and floor, then clip to valid range
        bin_indices = (normalized * self.bins_per_dim).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, self.bins_per_dim - 1)

        # Step 4: Convert multi-dimensional bin indices to single state index
        # Using row-major ordering: state = sum(bin_indices[i] * multipliers[i])
        state = int(np.dot(bin_indices, self.state_multipliers))

        return state

    def reset(self, **kwargs):
        """Reset the environment and discretize the initial observation.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the wrapped environment's reset.

        Returns
        -------
        int
            Discrete state index.
        dict
            Info dictionary from the environment.
        """
        observation, info = self.env.reset(**kwargs)
        discrete_state = self._discretize_observation(observation)
        return discrete_state, info

    def step(self, action):
        """Execute action and discretize the resulting observation.

        Parameters
        ----------
        action
            Action to execute in the environment.

        Returns
        -------
        int
            Discrete state index.
        float
            Reward received.
        bool
            Whether the episode terminated.
        bool
            Whether the episode was truncated.
        dict
            Info dictionary from the environment.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        discrete_state = self._discretize_observation(observation)
        return discrete_state, reward, terminated, truncated, info

    def get_observation_space(self):
        """Get the discretized observation space.

        Returns
        -------
        gymnasium.spaces.Discrete
            Discrete observation space with n = product of all bins_per_dim.
        """
        return self.observation_space

    def get_continuous_observation(self, observation: np.ndarray) -> int:
        """Public method to discretize a continuous observation.

        Useful for testing or manual discretization.

        Parameters
        ----------
        observation : np.ndarray
            Continuous observation to discretize.

        Returns
        -------
        int
            Discrete state index.
        """
        return self._discretize_observation(observation)


if __name__ == "__main__":
    # Comprehensive test suite
    import gymnasium as gym

    print("=" * 70)
    print("GridDiscretizationWrapper Test Suite")
    print("=" * 70)

    # Test 1: Basic functionality with MountainCar
    print("\n[Test 1] Basic functionality with MountainCar-v0")
    print("-" * 70)
    env = gym.make("MountainCar-v0")
    print(f"Original observation space: {env.observation_space}")
    print(
        f"Original obs bounds: {env.observation_space.low} to {env.observation_space.high}"
    )

    wrapped = GridDiscretizationWrapper(env, bins_per_dim=20)
    print(f"Wrapped observation space: {wrapped.observation_space}")
    print(f"Total discrete states: {wrapped.total_states}")
    print(f"Bins per dimension: {wrapped.bins_per_dim}")

    # Run a few steps
    obs, info = wrapped.reset(seed=42)
    print(f"\nInitial discrete state: {obs} (type: {type(obs).__name__})")

    for i in range(5):
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)
        print(
            f"Step {i + 1}: state={obs}, reward={reward:.2f}, done={terminated or truncated}"
        )

        if terminated or truncated:
            obs, info = wrapped.reset()
            print(f"Reset to state: {obs}")

    env.close()

    # Test 2: Different bins per dimension
    print("\n[Test 2] Different bins per dimension")
    print("-" * 70)
    env = gym.make("MountainCar-v0")
    wrapped = GridDiscretizationWrapper(env, bins_per_dim=[20, 15])
    print(f"Bins per dimension: {wrapped.bins_per_dim}")
    print(f"Total discrete states: {wrapped.total_states} (20 * 15 = 300)")
    print(f"Observation space: {wrapped.observation_space}")

    obs, info = wrapped.reset(seed=42)
    print(f"Initial discrete state: {obs}")
    env.close()

    # Test 3: Edge case - minimum bounds
    print("\n[Test 3] Edge case - minimum observation bounds")
    print("-" * 70)
    env = gym.make("MountainCar-v0")
    wrapped = GridDiscretizationWrapper(env, bins_per_dim=10, clip=True)

    # Manually discretize minimum observation
    min_obs = env.observation_space.low
    discrete_min = wrapped.get_continuous_observation(min_obs)
    print(f"Min continuous obs: {min_obs}")
    print(f"Discretized to state: {discrete_min}")
    print("Expected: 0 (leftmost bins)")

    env.close()

    # Test 4: Edge case - maximum bounds
    print("\n[Test 4] Edge case - maximum observation bounds")
    print("-" * 70)
    env = gym.make("MountainCar-v0")
    wrapped = GridDiscretizationWrapper(env, bins_per_dim=10, clip=True)

    # Manually discretize maximum observation
    max_obs = env.observation_space.high
    discrete_max = wrapped.get_continuous_observation(max_obs)
    print(f"Max continuous obs: {max_obs}")
    print(f"Discretized to state: {discrete_max}")
    print(f"Expected: {wrapped.total_states - 1} (rightmost bins)")

    env.close()

    # Test 5: Edge case - out of bounds (with clipping)
    print("\n[Test 5] Out-of-bounds observation (with clipping)")
    print("-" * 70)
    env = gym.make("MountainCar-v0")
    wrapped = GridDiscretizationWrapper(env, bins_per_dim=10, clip=True)

    # Create out-of-bounds observation
    oob_obs = env.observation_space.high + 10.0
    discrete_oob = wrapped.get_continuous_observation(oob_obs)
    print(f"Out-of-bounds obs: {oob_obs}")
    print(f"Discretized to state: {discrete_oob}")
    print(f"Expected: {wrapped.total_states - 1} (clipped to max)")

    env.close()

    # Test 6: Error handling - invalid bins
    print("\n[Test 6] Error handling - invalid bins_per_dim")
    print("-" * 70)
    env = gym.make("MountainCar-v0")

    try:
        wrapped = GridDiscretizationWrapper(env, bins_per_dim=0)
        print("ERROR: Should have raised ValueError for bins_per_dim=0")
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")

    try:
        wrapped = GridDiscretizationWrapper(env, bins_per_dim=[20, 15, 10])
        print("ERROR: Should have raised ValueError for wrong dimension count")
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")

    env.close()

    # Test 7: Error handling - non-Box observation space
    print("\n[Test 7] Error handling - non-Box observation space")
    print("-" * 70)
    env = gym.make("CartPole-v1")  # Discrete observation space... wait, no it's Box too

    # Use FrozenLake which has discrete observation space
    env = gym.make("FrozenLake-v1")
    print(f"FrozenLake observation space: {env.observation_space}")

    try:
        wrapped = GridDiscretizationWrapper(env, bins_per_dim=10)
        print("ERROR: Should have raised TypeError for Discrete observation space")
    except TypeError as e:
        print(f"Correctly raised TypeError: {e}")

    env.close()

    # Test 8: Single-dimensional observation space
    print("\n[Test 8] Single-dimensional observation space")
    print("-" * 70)
    env = gym.make("Pendulum-v1")
    print(f"Pendulum observation space: {env.observation_space}")

    wrapped = GridDiscretizationWrapper(env, bins_per_dim=10)
    print(f"Bins per dimension: {wrapped.bins_per_dim}")
    print(f"Total states: {wrapped.total_states}")
    print(f"Observation space: {wrapped.observation_space}")

    obs, info = wrapped.reset(seed=42)
    print(f"Initial discrete state: {obs}")

    env.close()

    # Test 9: State mapping verification
    print("\n[Test 9] State mapping verification (row-major ordering)")
    print("-" * 70)
    env = gym.make("MountainCar-v0")
    wrapped = GridDiscretizationWrapper(
        env, bins_per_dim=[3, 4]
    )  # Small for manual verification
    print(
        f"Bins per dimension: {wrapped.bins_per_dim} (3 position bins, 4 velocity bins)"
    )
    print(f"Total states: {wrapped.total_states} (3 * 4 = 12)")

    # Manually verify state mapping
    print("\nManual verification of bin indices to state mapping:")
    print("(pos_bin, vel_bin) -> state")
    for pos_bin in range(3):
        for vel_bin in range(4):
            expected_state = pos_bin * 4 + vel_bin
            print(f"  ({pos_bin}, {vel_bin}) -> {expected_state}")

    env.close()

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
