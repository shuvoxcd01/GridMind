from gridmind.value_estimators.base_nn_estimator import BaseNNEstimator


class NNValueEstimatorLinear(BaseNNEstimator):
    def __init__(self, observation_shape: tuple, use_bias: bool = False):
        super().__init__(
            observation_shape, num_hidden_layers=0, num_outputs=1, use_bias=use_bias
        )
