import gymnasium as gym


class IdleAgentTruncationWrapper(gym.Wrapper):
    def __init__(self, env, max_idle_frames=1000, max_repeated_actions=250):
        super().__init__(env)
        self.max_idle_frames = max_idle_frames
        self.max_repeated_actions = max_repeated_actions
        self.reset_tracking()

    def reset_tracking(self):
        self.idle_counter = 0
        self.last_action = None
        self.repeat_counter = 0

    def reset(self, **kwargs):
        self.reset_tracking()
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # Track repeated actions
        if action == self.last_action:
            self.repeat_counter += 1
        else:
            self.repeat_counter = 0
        self.last_action = action

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track idle time (no reward)
        if reward == 0:
            self.idle_counter += 1
        else:
            self.idle_counter = 0

        # Trigger early truncation if idle too long or repeating
        early_truncation = False
        if self.idle_counter >= self.max_idle_frames:
            early_truncation = True
            info["truncation_reason"] = "idle_timeout"
        elif self.repeat_counter >= self.max_repeated_actions:
            early_truncation = True
            info["truncation_reason"] = "repeated_action"

        if early_truncation:
            truncated = True
            terminated = False  # It didn't lose the game, just got stuck

        return obs, reward, terminated, truncated, info
