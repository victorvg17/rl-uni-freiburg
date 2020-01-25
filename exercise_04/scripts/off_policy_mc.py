from collections import defaultdict
import numpy as np

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

    def policy_fn(state):
        # pass
        # Implement this!
        action_prob = np.zeros_like(Q[state], dtype=float)
        greedy_action = np.argmax(Q[state])
        action_prob[greedy_action] = 1.0
        return action_prob

    return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
    Monte Carlo Control Off-Policy Control using Importance Sampling.
    Finds an optimal greedy policy.

    Args:
    env: environment.
    num_episodes: Nubmer of episodes to sample.
    behavior_policy: The behavior to follow while generating episodes.
        A function that given an observation returns a vector of probabilities for each action.
    discount_factor: Lambda discount factor.

    Returns:
    A tuple (Q, policy).
    Q is a dictionary mapping state -> action values.
    policy is a function that takes an observation as an argument and returns
    action probabilities. This is the optimal greedy policy.
    """

    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.nA))
    C = defaultdict(lambda: np.zeros(env.nA))

    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)
    episode_gen = []  #tuples of form: [state, action, reward]

    for e in range(num_episodes):
        state = env.reset()  #gives observation as the initial state.
        for i in range(100):  #each episode consider of the length 100
            next_action = np.argmax(behavior_policy(state))
            next_state, reward, done, _ = env.step(next_action)
            episode_gen.append([next_state, reward, done])
            state = next_state
            if done:
                break

    value_state = 0
    weight = 1
    for i in range(np.shape(episode_gen)[0]-1, -1, -1): #loop for each step of episode
        state, action, reward = episode_gen[i]
        value_state = value_state + discount_factor*reward
        C[state, action] = C[state, action] + weight
        Q[state, action] = Q[state, action] + (weight/C[state, action]) * (value_state - Q[state, action])
        action_greedy = np.argmax(target_policy(state))
        if (action_greedy != action):
            break
        weight = weight/behavior_policy(state)[action]
    # Implement this!

    return Q, target_policy
