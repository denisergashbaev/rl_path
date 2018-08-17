import matplotlib
import sys

from collections import defaultdict

if "../../" not in sys.path:
  sys.path.append("../../")
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')
env = BlackjackEnv()


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final value function
    V = defaultdict(float)

    for ep in range(num_episodes+1):
        # Print out which episode we're on, useful for debugging.
        if ep % 1000 == 0:
            print("\rEpisode {}/{}.".format(ep, num_episodes), end="")

        state = env.reset()
        # uncomment for some resemblance of reproducibility
        # state = (17, 10, False)
        done = False
        s_a_r = []
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            s_a_r.append((state, action, reward))
            state = next_state

            g = 0
            states = [x[0] for x in s_a_r]
            for idx, item in enumerate(reversed(s_a_r)):
                s, _, r = item
                g = discount_factor * g + r
                if s not in states[:len(s_a_r) - 1 - idx]:
                    returns_sum[s] += g
                    returns_count[s] += 1
                    V[s] = returns_sum[s] / returns_count[s]
    return V


def sample_policy(observation):
    """
    A policy that sticks if the player score is > 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")