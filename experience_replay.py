from collections import deque
import random

class ReplayMemory():
    def __init__(self, capacity, seed=None):
        """Initialize the experience replay buffer.

        Args:
            capacity (int): Maximum number of experiences to store in the buffer.
        """
        self.capacity = capacity
        self.buffer = deque([], maxlen=capacity)

        if seed is not None:
            random.seed(seed)

    def add(self, experience):
        """Add a new experience to the buffer.

        Args:
            experience (tuple): A tuple representing the experience (state, action, reward, next_state, done).
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)