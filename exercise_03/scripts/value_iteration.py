import numpy as np

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
    env: OpenAI environment. env.P represents the transition probabilities of the environment.
    theta: Stopping threshold. If the value of all states changes less than theta
      in one iteration we are done.
    discount_factor: lambda time discount factor.

    Returns:
    A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Get the best action and the value of the best action.
            A = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, ns, r, _ in env.P[s][a]:
                    A[a] += prob * (r + discount_factor * V[ns])
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function
            V[s] = best_action_value
        # Check if we can stop
        if delta < theta:
            break

    for s in range(env.nS):
      action_prob_s = np.zeros(env.nA)
      for a in range(env.nA):
          [(p, next_state, reward, done)] = env.P[s][a]
          action_prob_s[a] = p*(reward + discount_factor*V[next_state])
      greedy_a = np.argmax(action_prob_s)
      # modify policy as per greedy policy
      policy[s] = np.eye(env.nA)[greedy_a]

    return policy, V
