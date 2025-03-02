from gridmind.value_estimators.base_nn_estimator import BaseNNEstimator


class ActionValueEstimator(BaseNNEstimator):
    def __init__(
        self,
        observation_shape: tuple,
        num_hidden_layers: int,
        num_actions: int,
        use_bias: bool = True,
    ):
        super().__init__(
            observation_shape,
            num_hidden_layers,
            num_outputs=num_actions,
            use_bias=use_bias,
        )
