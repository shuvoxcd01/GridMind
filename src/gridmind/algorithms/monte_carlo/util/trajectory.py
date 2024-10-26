class Trajectory:
    def __init__(self) -> None:
        self.state_actions = list()
        self.rewards = list()

    def record_step(self, state, action, reward):
        state_action = (state, action)
        self.state_actions.append(state_action)
        self.rewards.append(reward)

    def get_step(self, timestep:int):
        state_action = self.get_state_action(timestep)
        reward = self.get_reward(timestep+1)

        state = state_action[0]
        action = state_action[1]

        return state, action, reward

    def get_state_action(self, timestep: int):
        assert timestep < len(self.state_actions) and timestep >= 0
        state_action = self.state_actions[timestep]

        return state_action

    def get_reward(self, timestep: int):
        assert timestep <= len(self.rewards) and timestep > 0
        reward = self.rewards[timestep - 1]

        return reward

    def check_state_action_appearance_before_timestep(self, state_action , timestep):
        has_appeared = state_action in self.state_actions[:timestep]

        return has_appeared

    def get_trajectory_length(self):
        trajectory_len = len(self.state_actions)

        return trajectory_len

    def clear(self):
        self.state_actions = list()
        self.rewards = list()