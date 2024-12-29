from gridmind.estimators.base_nn_estimator import BaseNNEstimator


class NNValueEstimatorMultilayer(BaseNNEstimator):
    def __init__(
        self, observation_shape: tuple, num_hidden_layers: int, use_bias: bool = True
    ):
        super().__init__(
            observation_shape, num_hidden_layers, num_outputs=1, use_bias=use_bias
        )
