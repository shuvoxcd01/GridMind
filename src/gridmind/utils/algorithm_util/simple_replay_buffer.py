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

    def sample(self, batch_size: int = 1):
        """Sample a batch of experiences."""
        if batch_size > self.size():
            raise ValueError("Batch size is greater than buffer size.")

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminated, truncated = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(terminated),
            np.array(truncated),
        )

    def size(self):
        """Return current buffer size."""
        return len(self.buffer)

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


if __name__ == "__main__":
    buffer = SimpleReplayBuffer(None)
    buffer.store(np.array([0, 0, 0]), 0, 1, np.array([1, 1, 1]), False, False)
    buffer.store(np.array([2, 2, 2]), 1, 0.5, np.array([3, 3, 3]), True, False)
    batch = buffer.sample(2)
    print(batch)
    print(buffer.size())
    buffer.clear()
    print(buffer.size())
