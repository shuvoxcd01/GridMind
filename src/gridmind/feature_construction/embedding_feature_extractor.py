from typing import Any
import torch.nn as nn
import torch
import numpy as np


class EmbeddingFeatureExtractor:
    def __init__(self, embedding: nn.Embedding) -> None:
        self.embedding = embedding
        self.device = self.embedding.weight.device

    def __call__(self, input_index: int, *args: Any, **kwds: Any) -> Any:
        with torch.no_grad():
            if isinstance(input_index, int):
                input_index = torch.tensor(input_index)
            elif isinstance(input_index, np.ndarray):
                input_index = np.squeeze(input_index, axis=-1)
                input_index = torch.from_numpy(input_index)

            embedding_out = (
                self.embedding(input_index.to(self.device)).detach().cpu().numpy()
            )

        return embedding_out
