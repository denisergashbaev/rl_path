import gym
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
        done = False
        states = []
        while not done:
            action = policy(state)
            s = env.step(action)
            next_state, reward, done, _ = s
            states.append((state, action, reward))
            state = next_state

        for state_idx, s_a_r in enumerate(states):
            g = 0
            for s in states[state_idx:]:
                _, action, reward = s
                g += discount_factor * reward
            state, _, _ = s_a_r
            returns_sum[state] += g
            returns_count[state] += 1
            V[state] = returns_sum[state] / returns_count[state]

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