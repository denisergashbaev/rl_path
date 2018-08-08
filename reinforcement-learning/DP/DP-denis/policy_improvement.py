import numpy as np
import pprint
import sys
if "../../" not in sys.path:
  sys.path.append("../../")
from lib.envs.gridworld import GridworldEnv
pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

# Taken from Policy Evaluation Exercise!
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s_idx, v in enumerate(V):
            new_v = 0
            for a_idx in range(env.nA):
                pi_s_a = policy[s_idx][a_idx]
                transitions = env.P[s_idx][a_idx]
                for prob, next_state, reward, done in transitions:
                    temp = pi_s_a * prob * (reward + discount_factor * V[next_state])
                    new_v += temp
            delta = max(delta, abs(new_v - v))
            V[s_idx] = new_v
        if delta < theta:
            break
    return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        V = policy_eval_fn(policy, env)
        policy_stable = True
        for s_idx in range(env.nS):
            old_action = np.argmax(policy[s_idx])
            new_policy_vals = np.zeros(env.nA)
            for a_idx in range(env.nA):
                transitions = env.P[s_idx][a_idx]
                for prob, next_state, reward, done in transitions:
                    temp = prob * (reward + discount_factor * V[next_state])
                    new_policy_vals[a_idx] += temp
            new_policy = np.zeros(env.nA)
            new_action = np.argmax(new_policy_vals)
            new_policy[new_action] = 1
            if old_action != new_action:
                policy_stable = False
            policy[s_idx] = np.eye(env.nA)[new_action]

        if policy_stable:
            break

    return policy, V

policy, v = policy_improvement(env)
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

# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)