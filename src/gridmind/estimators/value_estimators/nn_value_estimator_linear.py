from estimators.value_estimators.nn_value_estimator import NNValueEstimator


class NNValueEstimatorLinear(NNValueEstimator):
    def __init__(self, observation_shape: tuple):
        super().__init__(observation_shape, num_hidden_layers=0)
