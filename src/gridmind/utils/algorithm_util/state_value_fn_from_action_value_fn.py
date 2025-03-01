from typing import Callable, List
from gridmind.policies.base_policy import BasePolicy


def get_state_value_fn(
    action_value_fn: Callable, policy: BasePolicy, actions: List
) -> Callable:
    """
    It is assumed that every action is possible in every state.
    If not policy should return 0 action probility for that action.
    """
    state_value_fn = lambda state: sum(
        [
            action_value_fn(state, action)
            * policy.get_action_probs(state=state, action=action)
            for action in actions
        ]
    )

    return state_value_fn
