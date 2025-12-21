from gridmind.policies.parameterized.actor_critic_policy import ActorCriticPolicy
from gridmind.policies.parameterized.atari.atari_policy import AtariPolicy


class AtaricActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_shape, num_actions, channel_first: bool = True):
        self.channel_first = channel_first
        if self.channel_first:
            self.channels, self.height, self.width = observation_shape
        else:
            self.height, self.width, self.channels = observation_shape

        super(AtaricActorCriticPolicy, self).__init__(
            observation_shape=observation_shape, num_actions=num_actions
        )

    def construct_actor_critic_networks(self):  # type: ignore
        actor = AtariPolicy(
            observation_shape=self.observation_shape,
            num_actions=self.num_actions,
            channel_first=self.channel_first,
        )
        critic = AtariPolicy(
            observation_shape=self.observation_shape,
            num_actions=1,
            channel_first=self.channel_first,
        )

        return actor, critic

    def get_value(self, x):
        value_logits = self.critic(x)
        value = value_logits.view(-1)
        return value
