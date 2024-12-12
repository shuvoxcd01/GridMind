from estimators.value_estimators.nn_value_estimator import NNValueEstimator


class NNValueEstimatorMultilayer(NNValueEstimator):
    def __init__(self, observation_shape: tuple, num_hidden_layers: int):
        super().__init__(observation_shape, num_hidden_layers)
