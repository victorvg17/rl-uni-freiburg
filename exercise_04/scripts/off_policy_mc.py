from collections import defaultdict
import numpy as np
import sys

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
      A = np.zeros_like(Q[observation], dtype=float) #probabilities of two actions
      a_greedy = np.argmax(Q[observation])
      A[a_greedy] = 1.0 # set probability of greedy action to 1 and other remains 0.
      return A
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

    # Implement this!
    for i_episode in range(1, num_episodes + 1):
        b = create_random_policy(env.nA)

        # generate an episode using b
        # episode is tuple of (state, action , reward)
        episode = []
        state = env.reset()
        for t in range(100):
            action_prob = b(state)
            action_b = np.random.choice(len(action_prob), p=action_prob)
            next_state, reward, done, _ = env.step(action_b)
            episode.append((state, action_b, reward))

            if done:
                break
            state = next_state

        G = 0
        W = 1
        for state, action, reward in reversed(episode):
            G = discount_factor*G + reward
            C[state][action] += W
            sampling_ratio = W/C[state][action]
            Q[state][action] += sampling_ratio*(G - Q[state][action])
            actionTargetPolicy = np.argmax(target_policy(state))
            if actionTargetPolicy != action:
                # exit inner loop and proceed to next episode
                # if target and b has diverged, no point of continuing in the episode
                break
            W = W*1.0/b(state)[action]

        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            # print(f"\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            print(f'episode: {i_episode}, G: {G}, W: {W}')
            sys.stdout.flush()

    return Q, target_policy
