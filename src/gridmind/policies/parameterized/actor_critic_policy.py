from gridmind.policies.base_policy import BasePolicy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticPolicy(nn.Module, BasePolicy):
    def __init__(self, env):
        nn.Module.__init__(self)
        BasePolicy.__init__(self)

        observation_shape = np.array(env.observation_space.shape).prod()
        num_actions = env.action_space.n

        self.critic = nn.Sequential(
            nn.Linear(observation_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(observation_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), self.critic(x)

    def get_action(self, state):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()

        return action.detach().cpu().item()

    def get_action_probs(self, state, action):
        logits = self.actor(state)
        dist = Categorical(logits=logits)

        action_prob = dist.probs[action]

        return action_prob

    def update(self, state, action):
        raise NotImplementedError
