"""
Prioritized Experience Replay Buffer for Deep Reinforcement Learning.

This implementation extends SimpleReplayBuffer and uses a sum-tree data structure
for efficient sampling with priorities, supporting proportional prioritization.

Reference: Schaul et al., "Prioritized Experience Replay" (2015)
"""

from typing import Optional, Tuple
import numpy as np
import random
from gridmind.utils.algorithm_util.simple_replay_buffer import SimpleReplayBuffer


class SumTree:
    """
    Sum Tree data structure for efficient priority-based sampling.

    The tree structure allows O(log n) updates and O(log n) sampling,
    which is crucial for large replay buffers.

    Structure:
    - Parent node value = sum of children values
    - Leaf nodes contain priorities
    - Data indices are mapped to leaf positions
    """

    def __init__(self, capacity: int):
        """
        Initialize the sum tree.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        # Tree has capacity leaf nodes and capacity-1 internal nodes
        # Total nodes = 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # Current write position in the circular buffer
        self.write_index = 0
        # Current number of stored experiences
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """
        Propagate priority change up the tree.

        When a leaf priority changes, we need to update all ancestor nodes
        by adding the difference to maintain the sum property.

        Args:
            idx: Index of the changed leaf node
            change: Change in priority value
        """
        parent = (idx - 1) // 2

        self.tree[parent] += change

        # Continue propagating up the tree until root
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """
        Retrieve the leaf index corresponding to a priority value.

        This performs a binary search down the tree to find the leaf
        whose cumulative priority range contains the value s.

        Args:
            idx: Current node index (start with root)
            s: Target cumulative priority value

        Returns:
            Leaf index corresponding to the priority value
        """
        left = 2 * idx + 1
        right = left + 1

        # If we're at a leaf node, return its index
        if left >= len(self.tree):
            return idx

        # Go left if s is in the left subtree's range
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        # Otherwise go right (subtract left subtree's sum from s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return the total sum of all priorities (root node value)."""
        return self.tree[0]

    def add(self, priority: float, data_idx: int):
        """
        Add new experience priority.

        Args:
            priority: Priority value for this experience
            data_idx: Index in the data buffer
        """
        # Calculate tree index for this write position
        # Leaf nodes start at index (capacity - 1)
        tree_idx = data_idx + self.capacity - 1

        # Update the tree with the new priority
        self.update(tree_idx, priority)

        # Track number of entries
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        """
        Update priority for a specific tree index.

        Args:
            idx: Tree index to update
            priority: New priority value
        """
        # Calculate the change in priority
        change = priority - self.tree[idx]

        # Update this node
        self.tree[idx] = priority

        # Propagate the change up to the root
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, int]:
        """
        Get data index and priority for a cumulative priority value.

        Args:
            s: Target cumulative priority value

        Returns:
            Tuple of (tree_index, priority, data_index)
        """
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1

        return (tree_idx, self.tree[tree_idx], data_idx)


class PrioritizedReplayBuffer(SimpleReplayBuffer):
    """
    Prioritized Experience Replay Buffer extending SimpleReplayBuffer.

    This buffer samples experiences based on their TD-error, allowing
    the agent to learn more from surprising transitions. Includes
    importance sampling weights to correct for the bias introduced
    by non-uniform sampling.

    Key features:
    - Proportional prioritization: P(i) ∝ p_i^α
    - Importance sampling weights: w_i = (N * P(i))^(-β)
    - Efficient O(log n) sampling and updates via sum-tree
    - Handles both absolute and epsilon-based priorities
    - Extends SimpleReplayBuffer for compatibility
    """

    def __init__(
        self,
        capacity: Optional[int] = None,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
        max_priority: float = 1.0,
    ):
        """
        Initialize the prioritized replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            alpha: Controls how much prioritization is used (0 = uniform, 1 = full prioritization)
            beta_start: Initial value for importance sampling weight exponent
            beta_frames: Number of frames over which to anneal beta to 1
            epsilon: Small constant added to priorities to ensure non-zero probability
            max_priority: Initial priority for new experiences
        """
        if capacity is None:
            raise ValueError("Capacity must be specified for PrioritizedReplayBuffer")

        super().__init__(capacity=capacity)
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.max_priority = max_priority
        self.frame = 1

    def _get_priority(self, td_error: float) -> float:
        """
        Convert TD-error to priority.

        Priority is based on the absolute TD-error plus epsilon to ensure
        all transitions have non-zero probability of being sampled.

        Args:
            td_error: Temporal difference error

        Returns:
            Priority value
        """
        return (abs(td_error) + self.epsilon) ** self.alpha

    def store(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        truncated,
        td_error: Optional[float] = None,
    ):
        """
        Store a transition in the buffer with priority.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            td_error: TD-error for priority calculation (if None, uses max_priority)
        """
        # Get the index where this will be stored
        data_idx = (
            len(self.buffer)
            if len(self.buffer) < self.capacity
            else self.tree.write_index
        )

        # Use max priority for new experiences if TD-error not provided
        # This ensures new experiences get sampled at least once
        if td_error is None:
            priority = self.max_priority
        else:
            priority = self._get_priority(td_error)
            # Update max priority if this is larger
            self.max_priority = max(self.max_priority, priority)

        # Store in parent's buffer
        super().store(state, action, reward, next_state, terminated, truncated)

        # Add priority to tree
        self.tree.add(priority, data_idx)

        # Update write index for tree
        self.tree.write_index = (self.tree.write_index + 1) % self.capacity

    def sample(
        self,
        batch_size: int = 1,
        sequential: bool = False,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Sample a batch of experiences with importance sampling weights.

        The buffer is divided into batch_size segments, and one sample is drawn
        from each segment. This stratified sampling ensures coverage of the
        priority distribution.

        Args:
            batch_size: Number of experiences to sample
            sequential: Ignored for prioritized sampling (always uses priority-based sampling)

        Returns:
            Tuple of (states, actions, rewards, next_states, terminated, truncated,
                     importance_weights, tree_indices)
        """
        if sequential:
            # Fall back to parent's sequential sampling without priorities
            states, actions, rewards, next_states, terminated, truncated = (
                super().sample(batch_size, sequential=True)
            )
            # Return uniform weights and dummy indices for sequential sampling
            importance_weights = np.ones(batch_size, dtype=np.float32)
            tree_indices = np.zeros(batch_size, dtype=np.int32)
            return (
                states,
                actions,
                rewards,
                next_states,
                terminated,
                truncated,
                importance_weights,
                tree_indices,
            )

        if batch_size > self.size():
            raise ValueError(
                f"Batch size ({batch_size}) is greater than buffer size ({self.size()})."
            )

        batch = []
        tree_indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)

        # Divide priority range into batch_size segments
        segment_size = self.tree.total() / batch_size

        # Update beta (annealing for importance sampling)
        self.beta = min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames,
        )

        for i in range(batch_size):
            # Sample uniformly from this segment
            a = segment_size * i
            b = segment_size * (i + 1)
            value = random.uniform(a, b)

            # Get experience corresponding to this priority value
            tree_idx, priority, data_idx = self.tree.get(value)

            # Get data from parent's buffer
            batch.append(self.buffer[data_idx])
            tree_indices[i] = tree_idx
            priorities[i] = priority

        # Calculate importance sampling weights
        # w_i = (N * P(i))^(-β) / max_w
        sampling_probabilities = priorities / self.tree.total()
        importance_weights = (self.size() * sampling_probabilities) ** (-self.beta)

        # Normalize weights by max weight for stability
        importance_weights /= importance_weights.max()

        # Unpack batch
        states, actions, rewards, next_states, terminated, truncated = zip(*batch)

        # Convert to numpy arrays
        states_arr = np.array(states)
        actions_arr = np.array(actions)
        rewards_arr = np.array(rewards)
        next_states_arr = np.array(next_states)
        terminated_arr = np.array(terminated)
        truncated_arr = np.array(truncated)

        # Ensure proper shapes for 1D observations
        if states_arr.ndim == 1:
            states_arr = states_arr.reshape(-1, 1)
        if next_states_arr.ndim == 1:
            next_states_arr = next_states_arr.reshape(-1, 1)

        self.frame += 1

        return (
            states_arr,
            actions_arr,
            rewards_arr,
            next_states_arr,
            terminated_arr,
            truncated_arr,
            importance_weights,
            tree_indices,
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities for sampled experiences based on new TD-errors.

        This is called after training on a batch to update priorities based
        on the latest TD-errors.

        Args:
            tree_indices: Tree indices of the sampled experiences
            td_errors: New TD-errors for these experiences
        """
        for idx, td_error in zip(tree_indices, td_errors):
            priority = self._get_priority(td_error)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def clear(self):
        """Clear the buffer and reset to initial state."""
        super().clear()
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self.frame = 1


if __name__ == "__main__":
    # Test the prioritized replay buffer
    buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta_start=0.4)

    # Add some experiences with different TD-errors
    print("Adding experiences with varying TD-errors...")
    for i in range(50):
        td_error = np.random.random() * 10  # Random TD-errors
        buffer.store(
            state=np.array([i, i]),
            action=i % 4,
            reward=float(i),
            next_state=np.array([i + 1, i + 1]),
            terminated=False,
            truncated=False,
            td_error=td_error,
        )

    print(f"Buffer size: {buffer.size()}")

    # Sample a batch
    print("\nSampling a batch of 32 experiences...")
    (states, actions, rewards, next_states, terminated, truncated, weights, indices) = (
        buffer.sample(32)
    )

    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Importance weights shape: {weights.shape}")
    print(f"Importance weights (first 5): {weights[:5]}")
    print(f"Tree indices (first 5): {indices[:5]}")

    # Update priorities based on new TD-errors
    print("\nUpdating priorities with new TD-errors...")
    new_td_errors = np.random.random(32) * 5
    buffer.update_priorities(indices, new_td_errors)

    # Sample again to see if priorities changed
    print("\nSampling again after priority update...")
    (
        states2,
        actions2,
        rewards2,
        next_states2,
        terminated2,
        truncated2,
        weights2,
        indices2,
    ) = buffer.sample(32)

    print(f"New importance weights (first 5): {weights2[:5]}")
    print(f"New tree indices (first 5): {indices2[:5]}")

    # Test sequential sampling (falls back to uniform)
    print("\nTesting sequential sampling (should fall back to uniform)...")
    (
        states3,
        actions3,
        rewards3,
        next_states3,
        terminated3,
        truncated3,
        weights3,
        indices3,
    ) = buffer.sample(10, sequential=True)
    print(f"Sequential sampling weights (all should be 1.0): {weights3}")

    print("\nPrioritized replay buffer test completed successfully!")
