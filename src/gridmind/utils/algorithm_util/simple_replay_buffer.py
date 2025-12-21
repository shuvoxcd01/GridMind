from typing import Optional
import numpy as np
import random
from collections import deque


class SimpleReplayBuffer:
    def __init__(self, capacity: Optional[int] = None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, terminated, truncated):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, terminated, truncated))

    def extend(self, replay_buffer: "SimpleReplayBuffer"):
        """Extend the buffer with another buffer."""
        self.buffer.extend(replay_buffer.buffer)

    def sample(
        self,
        batch_size: int = 1,
        sequential: bool = False,
    ):
        """Sample a batch of experiences."""
        if batch_size > self.size():
            raise ValueError("Batch size is greater than buffer size.")

        if sequential:
            start_idx = random.randint(0, self.size() - batch_size)
            batch = [self.buffer[i] for i in range(start_idx, start_idx + batch_size)]
        else:
            batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, terminated, truncated = zip(*batch)

        states_arr = np.array(states)
        actions_arr = np.array(actions)
        rewards_arr = np.array(rewards)
        next_states_arr = np.array(next_states)
        terminated_arr = np.array(terminated)
        truncated_arr = np.array(truncated)

        if states_arr.ndim == 1:
            states_arr = states_arr.reshape(-1, 1)
        if next_states_arr.ndim == 1:
            next_states_arr = next_states_arr.reshape(-1, 1)

        return (
            states_arr,
            actions_arr,
            rewards_arr,
            next_states_arr,
            terminated_arr,
            truncated_arr,
        )

    def size(self):
        """Return current buffer size."""
        return len(self.buffer)

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self):
        """Return the current buffer size."""
        return self.size()

    def pop(self, num_elements: int = 1):
        """Pop elements from the buffer."""
        if num_elements > self.size():
            raise ValueError("Number of elements to pop is greater than buffer size.")
        return [self.buffer.popleft() for _ in range(num_elements)]


if __name__ == "__main__":
    buffer = SimpleReplayBuffer(None)
    buffer.store(np.array([0, 0, 0]), 0, 1, np.array([1, 1, 1]), False, False)
    buffer.store(np.array([2, 2, 2]), 1, 0.5, np.array([3, 3, 3]), True, False)
    batch = buffer.sample(2)
    print(batch)
    print(buffer.size())
    batch_2 = buffer.sample(2, sequential=True)
    print(batch_2)
    buffer.clear()
    print(buffer.size())
