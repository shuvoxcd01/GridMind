import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(
        self,
        num_states: int = 500,
        num_actions: int = 6,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_states, embedding_dim=embedding_dim
        )
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: Tensor of shape (batch_size,) with discrete state indices
        returns: Q-values for all actions, shape (batch_size, num_actions)
        """
        state = state.int()
        x = self.embedding(state)  # (batch_size, embedding_dim)
        q_values = self.fc(x)  # (batch_size, num_actions)
        return q_values
