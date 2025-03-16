import numbers
import random
from typing import Callable, Union
from gridmind.algorithms.base_learning_algorithm import BaseLearningAlgorithm
from gridmind.policies.parameterized.actor_critic_policy import ActorCriticPolicy
from gridmind.utils.algorithm_util.trajectory import Trajectory
from gymnasium import Env
from gymnasium.vector.vector_env import VectorEnv
import torch
from tqdm import trange
from torch.optim.adam import Adam

from src.gridmind.utils.performance_evaluation.basic_performance_evaluator import BasicPerformanceEvaluator

import logging

logging.basicConfig(level=logging.DEBUG)

class PPO(BaseLearningAlgorithm):
    def __init__(self,
        env:Env,
        num_actions: int,
        policy: ActorCriticPolicy,
        policy_step_size: float = 0.0001,
        value_step_size: float = 0.001,
        discount_factor: float = 1.0,
        lambda_:float =0.95,
        feature_constructor: Callable = None,
        clip_grads: bool = True,
        grad_clip_value: float = 1.0,):

        super().__init__("ProximalPolicyOptimization", env)
        self.policy = policy
        self.policy_step_size = policy_step_size
        self.value_step_size = value_step_size
        self.discount_factor = discount_factor
        self.clip_grads = clip_grads
        self.grad_clip_value = grad_clip_value
        self.minimum_timesteps = 500
        self.minibatch_size = 16
        self.lambda_ = lambda_
        self.epsilon = 0.2
        self.optimizer = Adam(self.policy.parameters())

        self.feature_constructor = feature_constructor
        observation_shape = (
            self.env.observation_space.shape
            if feature_constructor is None
            else self._determine_observation_shape()
        )

        self.num_actions = num_actions
    

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

    def _get_policy(self):
        return self.policy

    def set_policy(self, policy, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _create_minibatches_generator(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def _train(self, num_episodes:int, prediction_only:bool=False):
        if prediction_only:
            raise Exception("This is a control-only implementation.")
        
        trajectory = Trajectory()
        episode_count = 0

        while episode_count < num_episodes:
            observations = []
            actions = []
            rewards = []
            value_preds = []
            deltas = []
            advantages = []
            log_actionprobs = []

            step_count = 0

            with torch.no_grad():
                while step_count < self.minimum_timesteps:
                    episode_observations = []
                    episode_actions = []
                    episode_rewards = []
                    episode_value_preds = []
                    episode_log_actionprobs = []
                    
                    observation, info = self.env.reset()
                    done = False

                    while not done:
                        observation = self._preprocess(observation)
                        episode_observations.append(observation)
                        action, logprob,_, value = self.policy.get_action_and_value(observation)
                        episode_actions.append(action)
                        episode_log_actionprobs.append(logprob)
                        episode_value_preds.append(value)
                        action = action.detach().cpu().item()
                        next_observation, reward, terminated, truncated, _ = self.env.step(action)
                        step_count += 1
                        episode_rewards.append(reward)
                
                        observation = next_observation
                        done = terminated or truncated

                        if done:
                            next_state_value = 0.0
                            episode_deltas = []
                            episode_advantages = []

                            for reward, value in zip(reversed(episode_rewards), reversed(episode_value_preds)):
                                delta = reward + self.discount_factor * next_state_value - value
                                episode_deltas.append(delta)
                                
                                for ea_idx in range(len(episode_advantages)):
                                    episode_advantages[ea_idx] += self.discount_factor*self.lambda_*delta

                                episode_advantages.append(delta)
                                next_state_value = value
                            

                            deltas += reversed(episode_deltas)
                            advantages += reversed(episode_advantages)

                            observations += episode_observations
                            actions += episode_actions
                            rewards += episode_rewards
                            value_preds += episode_value_preds
                            log_actionprobs += episode_log_actionprobs

                            episode_count += 1
                            if episode_count >= num_episodes:
                                break
            
            for i in range(10):
                indices = list(range(len(observations)))
                random.shuffle(indices)

                for idx_batch in self._create_minibatches_generator(data=indices, batch_size=self.minibatch_size):
                    observations_batch = torch.stack([observations[idx] for idx in idx_batch])
                    actions_batch = torch.stack([actions[idx] for idx in idx_batch])
                    rewards_batch = torch.tensor([rewards[idx] for idx in idx_batch]).view(len(idx_batch),-1)
                    value_preds_batch = torch.stack([value_preds[idx] for idx in idx_batch])
                    deltas_batch = torch.stack([deltas[idx] for idx in idx_batch])
                    advantages_batch = torch.stack([advantages[idx] for idx in idx_batch])
                    log_actionprobs_batch = torch.stack([log_actionprobs[idx] for idx in idx_batch])
                    v_targ = value_preds_batch + advantages_batch


                    _, cur_logprob, _, cur_values = self.policy.get_action_and_value(observations_batch)

                    log_ratio = (log_actionprobs_batch - cur_logprob)
                    ratio = log_ratio.exp()

                    clipped_ratio = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)

                    clipped_surrogate_loss = - torch.mean(torch.min(ratio*advantages_batch, clipped_ratio*advantages_batch))
                    mse_loss = - torch.mean((v_targ - cur_values)**2)

                    total_loss = clipped_surrogate_loss + 0.5 * mse_loss


                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()



if __name__ == "__main__":
    from gridmind.policies.parameterized.actor_critic_policy import ActorCriticPolicy
    import gymnasium as gym


    env = gym.make("CartPole-v1")
    performance_evaluator = BasicPerformanceEvaluator(env= env)
    policy = ActorCriticPolicy(env)
    algorithm = PPO(env=env, num_actions=env.action_space.n, policy=policy)
    algorithm.register_performance_evaluator(performance_evaluator)

    algorithm.train(num_episodes=5000, prediction_only=False)




                


       
        

        

        

