from typing import Optional
from gridmind.policies.base_policy import BasePolicy
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import distance
from scipy.special import rel_entr  # for KL


class NoveltyUtils:
    @staticmethod
    def jensen_shannon_distance(p: ArrayLike, q: ArrayLike, eps: float = 1e-12) -> float:
        """
        Compute the Jensen–Shannon distance between two probability distributions.
        
        Parameters
        ----------
        p : ArrayLike
            First probability vector (will be normalized internally).
        q : ArrayLike
            Second probability vector.
        eps : float
            Small constant added for numerical stability.
            
        Returns
        -------
        float
            The Jensen–Shannon distance (sqrt(JS divergence)), in [0, 1].
        """
        
        # Convert to numpy arrays
        p = np.asarray(p, dtype=float) + eps
        q = np.asarray(q, dtype=float) + eps

        assert p.ndim == q.ndim == 1, "Input distributions must be 1-D arrays."
        assert p.shape == q.shape, "Input distributions must have the same shape."
        
        # Normalize so they sum to 1 but keep batch dimensions if present
        p /= p.sum(axis=-1, keepdims=True)
        q /= q.sum(axis=-1, keepdims=True)
        
        # Midpoint distribution
        m = 0.5 * (p + q)
        
        # KL divergence helper
        def kl(a, b):
            # Element-wise KL divergence but keep batch dimensions if present
            return np.sum(a * np.log(a / b), axis=-1, keepdims=True)
        
        # JS divergence
        js_div = 0.5 * kl(p, m) + 0.5 * kl(q, m)
        
        # Return distance (square root)
        distance = np.sqrt(js_div).squeeze().item()

        return distance
    

    @staticmethod
    def get_average_probability_distribution(distributions: ArrayLike) -> np.ndarray:
        """
        Compute the average probability distribution from a set of distributions.
        
        Parameters
        ----------
        distributions : ArrayLike
            Array of shape (N, D) where N is the number of distributions and D is the dimension.
            
        Returns
        -------
        np.ndarray
            The average probability distribution of shape (D,).
        """
        distributions = np.asarray(distributions, dtype=float)
        avg_distribution = np.mean(distributions, axis=0)
        avg_distribution /= np.sum(avg_distribution, axis=-1, keepdims=True)  # Normalize to sum to 1
        return avg_distribution

    @staticmethod
    def make_weights(m, gamma=0.95):
        w = np.array([gamma**i for i in range(m)])
        return w / w.sum()

    @staticmethod
    def policy_bc_on_observations(policy:BasePolicy, observations, weights:Optional[np.ndarray]=None):
        if weights is None:
            weights = np.ones(len(observations))
        
        parts = []
        for i, o in enumerate(observations):
            probs = policy.get_all_action_probabilities(o)            # (A,)
            parts.append(np.sqrt(weights[i]) * probs)
        return np.concatenate(parts)     # shape (m*A,)

    @staticmethod
    def weighted_js(p, q, eps=1e-12):
        m = 0.5*(p+q)
        return 0.5*(np.sum(rel_entr(p+eps, m+eps)) + np.sum(rel_entr(q+eps, m+eps)))

    @staticmethod
    def dist_weighted_js(policy1, policy2, observations, weights):
        total = 0.0
        for i,o in enumerate(observations):
            p = policy1(o); q = policy2(o)
            total += weights[i] * NoveltyUtils.weighted_js(p,q)
        return total


# if __name__ == "__main__":
#     # Example usage
#     p = [0.1, 0.9]
#     q = [0.7, 0.3]
#     r = [0.4, 0.6]
#     p_array = np.array([p,p])
#     q_array = np.array([q,q])
#     avg_distribution = NoveltyUtils.get_average_probability_distribution([p_array, q_array])
#     print(f"Average Distribution: {avg_distribution}")
#     distance = NoveltyUtils.jensen_shannon_distance(p_array, avg_distribution)
#     print(f"Jensen–Shannon distance\n: {distance}")
#     distance2 = NoveltyUtils.jensen_shannon_distance(q_array, avg_distribution)
#     print(f"Jensen–Shannon distance\n: {distance2}")
#     distance3 = NoveltyUtils.jensen_shannon_distance(np.array(r), avg_distribution)
#     print(f"Jensen–Shannon distance\n: {distance3}")

#     # example usage
#     m = 100
#     observations = sample_probe_set(m)   # your fixed probe observations
#     weights = NoveltyUtils.make_weights(m, gamma=0.95)

#     bc1 = NoveltyUtils.policy_bc_on_observations(pi1, observations, weights)
#     bc2 = NoveltyUtils.policy_bc_on_observations(pi2, observations, weights)
#     euclid = np.linalg.norm(bc1-bc2)
#     js = NoveltyUtils.dist_weighted_js(pi1, pi2, observations, weights)