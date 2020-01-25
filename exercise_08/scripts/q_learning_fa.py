import sys
import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from collections import namedtuple
import time
import matplotlib.pyplot as plt
import pandas as pd
from mountain_car import MountainCarEnv
import random

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def tt(ndarray):
  return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)

def soft_update(target, source, tau):
  #Implement this
  assert len(target) == len(source), "unequal legnth for target and source vectors"
  len_target = target.shape[0]
  for i in range(len_target):
      target[i] = target*(1 - tau) + source*tau
  return target
  # raise NotImplementedError("Implement a function to slowly update the parameters of target by the parameters of source with step size tau")

def hard_update(target, source):
  #Implement this
  assert len(target) == len(source), "unequal legnth for target and source vectors"
  len_target = target.shape[0]
  for i in range(len_target):
      target[i] = source[i]
  return target
  # raise NotImplementedError("Implement a function to completely overwrite the parameters of target by the parameters of source")

class Q(nn.Module):
  def __init__(self, state_dim, action_dim, non_linearity=F.relu, hidden_dim=50):
    super(Q, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, action_dim)
    self._non_linearity = non_linearity

  def forward(self, x):
    x = self._non_linearity(self.fc1(x))
    x = self._non_linearity(self.fc2(x))
    return self.fc3(x)

class ReplayBuffer:
  #Replay buffer for experience replay. Stores transitions.
  def __init__(self, max_size):
    self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
    self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
    self._size = 0
    self._max_size = max_size

  def add_transition(self, state, action, next_state, reward, done):
    # Implement this
    self._data.states.append(state) #append state to states
    self._data.actions.append(action) #append action to actions
    self._data.next_states.append(next_state) #append next_state to next_states
    self._data.rewards.append(reward) #append reward to rewards
    self._data.terminal_flags.append(done) #append done to terminal_flags
    # self._data.append(state, action, next_state, reward, done)
    # raise NotImplementedError("Implement the method that adds a transition to the replay buffer")


  def random_next_batch(self, batch_size):
    """
    Args:
        batch_size: Size of batches to sample from the replay memory
        # Implement this
    """
    drawn_samples = random.sample(self._data, batch_size)
    # (state, action, next_state, reward, done)
    state_sampled, action_sampled, next_state_sampled, reward_sampled, done_sampled = map(np.array, drawn_samples)
    return state_sampled, action_sampled, next_state_sampled, reward_sampled, done_sampled
    # raise NotImplementedError("Implement the method that draws a random minibatch from the replay buffer")

class DQN:
    def __init__(self, state_dim, action_dim, gamma):
        self._q = Q(state_dim, action_dim)
        self._q_target = Q(state_dim, action_dim)

        # self._q.cuda()
        # self._q_target.cuda()

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._action_dim = action_dim

        self._replay_buffer = ReplayBuffer(1e6)

    def get_action(self, x, epsilon):
        u = np.argmax(self._q(tt(x)).cpu().detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
          return np.random.randint(self._action_dim)
        return u

    def _train_weights_step(self, input_target, output_target):
        self._q_optimizer.zero_grad()
        loss_q_val = self._loss_function(q_esti, q_target)
        loss_q_val.backward()
        self._q_optimizer.step()


    def train(self, episodes, time_steps, epsilon):
        stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))

        for e in range(episodes):
          print("%s/%s"%(e+1, episodes))
          s = env.reset()
          for t in range(time_steps):
            a = self.get_action(s, epsilon)
            ns, r, d, _ = env.step(a)

            stats.episode_rewards[e] += r
            stats.episode_lengths[e] = t

            # Implement this
            # store transitions (state, action, next_state, reward, done) in replay memory
            self._replay_buffer.add_transition(s, a, ns, r, d)
            s, a, ns, r, d = self._replay_buffer.random_next_batch(batch_size=5)

            # importing the training part from sarsa_fa.py
            q_esti = self._q(tt(s))
            q_target = q_esti.clone()

            if (d):
                q_target[a] = r
                _train_weights_step(input_target=q_esti, output_target=q_target)
                break

            # get the Q values for next state: ns

            q_estim_next = self._q(tt(ns))
            next_action = self.get_action(ns, epsilon)
            q_target[curr_action] = r + self._gamma * q_estim_next[next_action]

            _train_weights_step(input_target=q_esti, output_target=q_target)

            if d:
              break

            s = ns

        return stats

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
  # Plot the episode length over time
  fig1 = plt.figure(figsize=(10,5))
  plt.plot(stats.episode_lengths)
  plt.xlabel("Episode")
  plt.ylabel("Episode Length")
  plt.title("Episode Length over Time")
  fig1.savefig('episode_lengths.png')
  if noshow:
      plt.close(fig1)
  else:
      plt.show(fig1)

  # Plot the episode reward over time
  fig2 = plt.figure(figsize=(10,5))
  rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
  plt.plot(rewards_smoothed)
  plt.xlabel("Episode")
  plt.ylabel("Episode Reward (Smoothed)")
  plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
  fig2.savefig('reward.png')
  if noshow:
      plt.close(fig2)
  else:
      plt.show(fig2)

if __name__ == "__main__":
  env = MountainCarEnv() #gym.make("MountainCar-v0")
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.n
  dqn = DQN(state_dim, action_dim, gamma=0.99)

  episodes = 1000
  time_steps = 200
  epsilon = 0.2

  stats = dqn.train(episodes, time_steps, epsilon)

  plot_episode_stats(stats)

  for _ in range(5):
    s = env.reset()
    for _ in range(200):
      env.render()
      a = dqn.get_action(s, epsilon)
      s, _, d, _ = env.step(a)
      if d:
        break
