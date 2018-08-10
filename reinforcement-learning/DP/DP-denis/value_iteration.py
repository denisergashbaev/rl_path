import numpy as np
import pprint
import sys
if "../../" not in sys.path:
  sys.path.append("../../")
from lib.envs.gridworld import GridworldEnv
pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    while True:
        diff = 0
        for s_idx in range(env.nS):
            old_v = V[s_idx]
            v_temp = np.zeros(env.nA)
            for a_idx in range(env.nA):
                for prob, next_state, reward, done in env.P[s_idx][a_idx]:
                    v_temp[a_idx] += prob * (reward + discount_factor * V[next_state])
            best_a_idx = np.argmax(v_temp)
            policy[s_idx] = np.eye(env.nA)[best_a_idx]
            V[s_idx] = v_temp[best_a_idx]
            diff = max(diff, abs(V[s_idx] - old_v))
        if diff < theta:
            break
    return policy, V


policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)