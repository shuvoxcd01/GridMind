from gridmind.policies.base_policy import BasePolicy
from gridmind.value_estimators.action_value_estimators.q_network import QNetwork
import torch


class QNetworkToStateValueEstimatorWrapper:
    def __init__(self, q_network: QNetwork, policy:BasePolicy):
        self.q_network = q_network
        self.policy = policy

    def forward(self, x):
        q_values = self.q_network.forward(x)
        with torch.no_grad():
            action_probs = self.policy.get_all_action_probabilities(x)
        state_value = torch.sum(action_probs * q_values, dim=-1)
        return state_value
    
    def set_q_network(self, q_network: QNetwork):
        self.q_network = q_network
    
    def set_policy(self, policy: BasePolicy):
        self.policy = policy
    

if __name__ == "__main__":
    from gridmind.policies.parameterized.discrete_action_mlp_policy import DiscreteActionMLPPolicy

    observation_shape = (4,)
    num_actions = 2

    q_network = QNetwork(observation_shape=observation_shape, num_hidden_layers=2, num_actions=num_actions)
    policy = DiscreteActionMLPPolicy(observation_shape=observation_shape, num_actions=num_actions)

    wrapper = QNetworkToStateValueEstimatorWrapper(q_network=q_network, policy=policy)

    sample_input = torch.randn(1, *observation_shape)
    state_value = wrapper.forward(sample_input)

    print("Estimated State Value:", state_value.item())
   