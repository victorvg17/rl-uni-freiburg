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
    def one_step_look_ahead(state, V):
        """
        helper function
        """
        action_val = np.zeros((env.nA, 1))
        for a in range(env.nA):
            [(prob, next_s, reward, done)] = env.P[state][a]
            action_val[a] = prob*(reward + discount_factor*V[next_s])
        return action_val

    V = np.zeros(env.nS)
    # TODO: Implement this!
    while True:
        delta = 0.0
        for s in range(env.nS):
            best_action_val = np.max(one_step_look_ahead(s, V))
            delta = max(delta, np.abs(best_action_val - V[s]))
            V[s] = best_action_val
        if (delta < theta):
            break
    # create a deterministic policy
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        best_action = one_step_look_ahead(state, V)
        best_action_val = np.argmax(best_action)
        policy[s, best_action_val] = 1.0
    return policy, V
