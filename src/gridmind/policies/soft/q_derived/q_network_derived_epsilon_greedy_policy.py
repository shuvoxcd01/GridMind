from typing import Optional

import numpy as np
import torch

from gridmind.policies.soft.q_derived.base_q_derived_soft_policy import (
    BaseQDerivedSoftPolicy,
)


class QNetworkDerivedEpsilonGreedyPolicy(BaseQDerivedSoftPolicy):
    def __init__(
        self,
        q_network: torch.nn.Module,
        num_actions: int,
        action_space=None,
        epsilon=0.1,
        allow_decay=True,
        epsilon_min=0.001,
        decay_rate=0.01,
    ):
        super().__init__(Q=q_network, epsilon=epsilon, num_actions=num_actions)
        self.action_space = action_space
        self.allow_decay = allow_decay
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.device = next(self.Q.parameters()).device

        assert 0 <= epsilon <= 1, "epsilon must be in rage 0 to 1."
        assert (
            num_actions == self.action_space.n
            if self.action_space is not None
            else True
        ), "Provided num_actions does not match with number of actions in the provided action_space."

    def get_network(self):
        return self.Q

    def set_network(self, network):
        self.Q = network
        self.device = next(self.Q.parameters()).device

    def update(self, state, action):
        raise Exception(
            "This policy is derived from q_network. Instead of directly updating the action to take in a state, please update the state-action value. Use update_q method instead."
        )

    def update_q(self, state, action, value: float):
        raise Exception(
            f"{self.__class__.__name__} does not support updating Q values directly."
        )

    def _get_greedy_action(self, state, action_mask: Optional[np.ndarray] = None):
        state = state.to(self.device)
        q_values = self.Q(state)

        if action_mask is not None:
            # Convert numpy mask to torch tensor and move to the same device as q_values
            action_mask_tensor = torch.tensor(
                action_mask, dtype=torch.bool, device=self.device
            )
            # Mask invalid actions with -inf
            masked_q_values = torch.where(
                action_mask_tensor,
                q_values,
                torch.tensor(-torch.inf, device=self.device),
            )
            action = torch.argmax(masked_q_values).cpu().detach().item()
        else:
            action = torch.argmax(q_values).cpu().detach().item()

        assert (
            action in self.action_space if self.action_space is not None else True
        ), "Action not in action space!!"

        return action

    def set_epsilon(self, value: float):
        if value < self.epsilon_min:
            self.logger.warning(
                "Trying to set epsilon value less than epsilon_min. Setting epsilon=epsilon_min"
            )
            value = self.epsilon_min

        super().set_epsilon(value)

    def decay_epsilon(self):
        if not self.allow_decay:
            self.logger.warning("Epsilon decay is not allowed.")
            return

        decayed_epsilon = self.epsilon - self.decay_rate

        if decayed_epsilon >= self.epsilon_min:
            self.set_epsilon(value=decayed_epsilon)
