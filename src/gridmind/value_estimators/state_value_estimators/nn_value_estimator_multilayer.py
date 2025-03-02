from gridmind.value_estimators.base_nn_estimator import BaseNNEstimator


class NNValueEstimatorMultilayer(BaseNNEstimator):
    def __init__(
        self,
        observation_shape: tuple,
        num_hidden_layers: int,
        use_bias: bool = True,
        in_features: int = 16,
        out_features: int = 16,
    ):
        super().__init__(
            observation_shape,
            num_hidden_layers,
            num_outputs=1,
            use_bias=use_bias,
            in_features=in_features,
            out_features=out_features,
        )
