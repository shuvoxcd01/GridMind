from gridmind.policies.parameterized.base_parameterized_policy import (
    BaseParameterizedPolicy,
)
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticPolicy(BaseParameterizedPolicy):
    def __init__(self, observation_shape, num_actions):
        super(ActorCriticPolicy, self).__init__(
            observation_shape=observation_shape, num_actions=num_actions
        )

        self.actor, self.critic = self.construct_actor_critic_networks()

    def construct_actor_critic_networks(self):
        input_size = int(np.prod(self.observation_shape))

        critic = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        actor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )

        return actor, critic

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action.squeeze()), dist.entropy(), self.critic(x)

    def get_action(self, state):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()

        return action.detach().cpu().item()

    def get_action_prob(self, state, action):
        logits = self.actor(state)
        dist = Categorical(logits=logits)

        action_prob = dist.probs[action]

        return action_prob

    def update(self, state, action):
        raise NotImplementedError
