from gridmind.wrappers.policy_wrappers.base_policy_wrapper import BasePolicyWrapper


class PreprocessedObservationPolicyWrapper(BasePolicyWrapper):
    def __init__(self, policy, preprocess_fn):
        super().__init__(policy)
        self.preprocess_fn = preprocess_fn

    def get_action(self, state):
        preprocessed_state = self.preprocess_fn(state)
        return self.policy.get_action(preprocessed_state)
    
    def get_action_probs(self, state, action):
        preprocessed_state = self.preprocess_fn(state)
        return self.policy.get_action_probs(preprocessed_state, action)
    
    def update(self, state, action):
        preprocessed_state = self.preprocess_fn(state)
        return self.policy.update(preprocessed_state, action)