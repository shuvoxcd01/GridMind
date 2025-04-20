
import numbers
from typing import List, Optional
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.policies.base_policy import BasePolicy
from gridmind.policies.parameterized.discrete_action_mlp_policy import DiscreteActionMLPPolicy
from gridmind.policies.random_policy import RandomPolicy
from gridmind.utils.algorithm_util.episode_collector import collect_episode
from gridmind.utils.algorithm_util.trajectory import Trajectory
from gridmind.utils.performance_evaluation.basic_performance_evaluator import BasicPerformanceEvaluator
from gridmind.value_estimators.state_value_estimators.nn_value_estimator_multilayer import NNValueEstimatorMultilayer
from gridmind.wrappers.policy_wrappers.epsilon_randomized_policy_wrapper import EpsilonRandomizedPolicyWrapper
from gymnasium import Env
import torch
from tqdm import trange



class ReinforceOffPolicyExperience(BaseLearningAlgorithm):
    def __init__(self,  env:Env, 
                trajectories:Optional[List[Trajectory]]=None,
                target_policy:Optional[DiscreteActionMLPPolicy]=None, 
                behavior_policy:Optional[BasePolicy]=None,
                value_estimator:Optional[NNValueEstimatorMultilayer]  = None,
                policy_step_size:float=0.0001,
                value_step_size:float=0.001,
                discount_factor:float=0.99, 
                feature_constructor=None, 
                grad_clip_value: float = 1.0,
                summary_dir:Optional[str] = None,
                write_summary:bool = True):
        
        super().__init__("ReinforceOffPolicy_research", env, summary_dir=summary_dir, write_summary=write_summary)
        self.target_policy = target_policy
        self.policy_step_size = policy_step_size
        self.value_step_size = value_step_size
        self.discount_factor = discount_factor
        self.feature_constructor = feature_constructor
        self.grad_clip_value = grad_clip_value
        self.trajectories = trajectories

        observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )

        self.num_actions = self.env.action_space.n

        self.target_policy = (
            target_policy
            if target_policy is not None
            else DiscreteActionMLPPolicy(
                observation_shape=observation_shape,
                num_actions=self.num_actions,
                num_hidden_layers=2,
            )
        )
        self.behavior_policy = behavior_policy if behavior_policy is not None else EpsilonRandomizedPolicyWrapper(policy=self.target_policy, num_actions=self.num_actions, epsilon=0.2)

        self.value_estimator = (
            value_estimator
            if value_estimator is not None
            else NNValueEstimatorMultilayer(
                observation_shape=observation_shape, num_hidden_layers=2
            )
        )
        self.global_step = 0
        self.max_importance_weight = 5.0

    def _determine_observation_shape(self):
        observation, _ = self.env.reset()
        
        features = self.feature_constructor(observation)

        shape = features.shape

        return shape

    def _preprocess(self, obs):
        if self.feature_constructor is not None:
            obs = self.feature_constructor(obs)

        if isinstance(obs, numbers.Number):
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        else:
            obs = torch.tensor(obs, dtype=torch.float32)

        return obs
   
    def _get_state_value_fn(self, force_functional_interface = True):
        raise NotImplementedError

    def _get_state_action_value_fn(self, force_functional_interface = True):
        raise NotImplementedError
    
    def get_value_estimator(self):
        return self.value_estimator
    
    def set_value_estimator(self, value_estimator):
        self.value_estimator = value_estimator

    def _get_policy(self):
        return self.target_policy

    def set_policy(self, policy:DiscreteActionMLPPolicy, update_behavior_policy:bool = False):
        self.target_policy = policy
        if update_behavior_policy:
            self.behavior_policy = EpsilonRandomizedPolicyWrapper(policy=policy, num_actions=self.num_actions, epsilon=0.2)

    def get_trajectories(self):
        return self.trajectories
    
    def set_trajectories(self, trajectories:List[Trajectory]):
        self.trajectories = trajectories

    def log_weights_and_grads(self, model, grads, step, prefix):
        """Logs weights and manually computed gradients to TensorBoard."""
        for (name, param), grad in zip(model.named_parameters(), grads):
            if grad is not None:
                self.summary_writer.add_histogram(f"Gradients/{prefix}/{name}", grad, step)
            self.summary_writer.add_histogram(f"Weights/{prefix}/{name}", param.data, step)


    def _train(self, num_episodes, prediction_only:bool = False):
        if prediction_only:
            raise NotImplementedError("Prediction only is not supported for Reinforce")
        
        assert self.trajectories is not None
        num_episodes_in_trajectories = len(self.trajectories)
        num_remaining_episodes = num_episodes - num_episodes_in_trajectories
        
        if num_remaining_episodes < 0:
            self.trajectories = self.trajectories[:num_episodes]
            num_remaining_episodes = 0
        
        
        for trajectory in self.trajectories:
            self._train_with_trajectory(trajectory)

        for _ in trange(num_remaining_episodes):
            trajectory = Trajectory()
            collect_episode(env=self.env, policy=self.behavior_policy, trajectory=trajectory, obs_preprocessor=self._preprocess, record_action_prob=True)
            self._train_with_trajectory(trajectory)


    def _train_with_trajectory(self, trajectory):
        discounted_return = 0.0
        discounted_returns = []
        for timestep in reversed(range(trajectory.get_trajectory_length())):
            reward = trajectory.get_reward(timestep+1)
            discounted_return = self.discount_factor * discounted_return + reward
            discounted_returns.append(discounted_return)

        discounted_returns = list(reversed(discounted_returns))
            
        importance_weight = 1.0

        for timestep in range(trajectory.get_trajectory_length()):
            obs, action, reward, info = trajectory.get_step_with_info(timestep)
            obs = self._preprocess(obs)
            discounted_return = discounted_returns[timestep]

            target_action_prob = self.target_policy.get_action_probs(obs, action)
            behavior_action_prob = info["action_prob"]
            value_pred = self.value_estimator(obs)
            
            importance_weight *= target_action_prob / behavior_action_prob
            importance_weight = min(importance_weight, self.max_importance_weight)
            
            if self.write_summary:
                self.summary_writer.add_scalar("Importance_Weight", importance_weight, self.global_step)

            assert importance_weight >= 0, "Importance weight must be non-negative"
                
            if importance_weight == 0:
                self.logger.debug("Importance weight is 0. Skipping the rest of the trajectory")
                break
                
            log_prob = torch.log(target_action_prob)
            delta = discounted_return - value_pred

            value_grads = torch.autograd.grad(
                    value_pred,
                    self.value_estimator.parameters(),
                )

                #self.logger.debug(f"Value grads: {value_grads}")

            
            policy_grads = torch.autograd.grad(
                    log_prob,
                    self.target_policy.parameters(),
                )

                #self.logger.debug(f"Policy grads: {policy_grads}")
            if timestep % 100 == 0 and self.write_summary:
                self.log_weights_and_grads(self.value_estimator, value_grads, self.global_step, "Value_Estimator")
                self.log_weights_and_grads(self.target_policy, policy_grads, self.global_step, "Target_Policy")

            with torch.no_grad():
                for param, grad in zip(
                        self.value_estimator.parameters(), value_grads
                    ):
                    param.copy_(param.data + importance_weight * self.value_step_size * delta * grad)
                
            with torch.no_grad():
                for param, grad in zip(
                        self.target_policy.parameters(), policy_grads
                    ):
                    param.copy_(param.data + importance_weight * self.policy_step_size * (self.discount_factor**timestep) * delta * grad)

            self.global_step += 1


if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium.wrappers import NormalizeReward
    env = gym.make("CartPole-v1")

    eval_env = gym.make("CartPole-v1", render_mode="rgb_array")

    performance_evaluator = BasicPerformanceEvaluator(env= eval_env, epoch_eval_interval=500)
    target_policy = DiscreteActionMLPPolicy(
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        num_hidden_layers=2,
    )
    behavior_policy = EpsilonRandomizedPolicyWrapper(policy=target_policy, num_actions=env.action_space.n, epsilon=0.2)
    # policy = ActorCriticPolicy(env)
    algorithm = ReinforceOffPolicyExperience(env=env, target_policy=target_policy, policy_step_size=0.0001)
    algorithm.register_performance_evaluator(performance_evaluator)

    algorithm.train(num_episodes=100000, prediction_only=False)