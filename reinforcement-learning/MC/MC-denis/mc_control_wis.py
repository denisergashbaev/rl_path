import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../../" not in sys.path:
  sys.path.append("../../")
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()


def create_random_policy(nA):
    """
    Creates a random policy function.

    Args:
        nA: Number of actions in the environment.

    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA

    def policy_fn(observation):
        return A

    return policy_fn


def create_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values.

    Args:
        Q: A dictionary that maps from state -> action values

    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """

    def policy_fn(observation):
        actions = Q[observation]
        n_actions = np.zeros(len(actions))
        idx = np.argmax(actions)
        n_actions[idx] = 1
        return n_actions
    return policy_fn

    return policy_fn


def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. This is the optimal greedy policy.
    """

    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)

    c = defaultdict(lambda: np.zeros(env.action_space.n))

    for ep_idx in range(num_episodes):
        if ep_idx % 1000 == 0:
            print('running episode {}/{}'.format(ep_idx, num_episodes), end='\n')

        # The policy we're following
        done = False
        s_a_r = []
        state = env.reset()
        while not done:
            beh_a = behavior_policy(state)
            a = np.random.choice(len(beh_a), p=beh_a)
            obs, reward, done, _ = env.step(a)
            s_a_r.append((state, a, reward))
            state = obs
        g = 0
        w = 1
        for idx, rev_s_a_r in enumerate(reversed(s_a_r)):
            s, a, r = rev_s_a_r
            g = discount_factor * g + r
            c[s][a] += w
            Q[s][a] += w / c[s][a] * (g - Q[s][a])
            targ_a = target_policy(s)
            if a != np.argmax(targ_a):
                break
            w = w * 1/behavior_policy(state)[a]
    return Q, target_policy


random_policy = create_random_policy(env.action_space.n)
Q, policy = mc_control_importance_sampling(env, num_episodes=5000000, behavior_policy=random_policy)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, action_values in Q.items():
    action_value = np.max(action_values)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")