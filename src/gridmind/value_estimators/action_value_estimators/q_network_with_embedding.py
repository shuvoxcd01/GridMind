from gridmind.value_estimators.base_nn_estimator import BaseNNEstimator
import torch.nn as nn

class QNetworkWithEmbedding(BaseNNEstimator):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim:int,
        num_hidden_layers: int,
        num_actions: int,
        use_bias: bool = True,
        
    ):
        super().__init__(
            observation_shape=(embedding_dim,),
            num_hidden_layers=num_hidden_layers,
            num_outputs=num_actions,
            use_bias=use_bias,
        )
        
        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def get_embedding(self):
        self.embedding_layer.eval()
        return self.embedding_layer

    def forward(self, x):
        self.embedding_layer.train()

        x = self.embedding_layer(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        out = self.linear_out(x)

        return out
