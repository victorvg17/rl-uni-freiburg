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

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def tt(ndarray):
  # Use this if you have CUDA supported GPU
  #return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
  return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)

class Q(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=50):
    super(Q, self).__init__()
    # Implement this
    self.hidden = nn.Linear(in_features=state_dim, out_features=hidden_dim)
    #adding relu layer for much needed non-linearity
    self.relu = nn.ReLU()
    self.output = nn.Linear(in_features=hidden_dim, out_features=action_dim)
    # raise NotImplementedError("Architecture of the Q-network is missing.")

  def forward(self, x):
    # Implement this
    x = self.hidden(x)
    x = self.relu(x)
    x = self.output(x)

    # raise NotImplementedError("Forward pass of the Q-network is missing.")
    return x

class SARSA:
  def __init__(self, state_dim, action_dim, gamma):
    self._q = Q(state_dim, action_dim)
    # Untoggle this if you have CUDA supported GPU
    # self._q.cuda()
    self._gamma = gamma
    self._loss_function = nn.MSELoss()
    self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0005)
    self._action_dim = action_dim

  def get_action(self, x, epsilon):
    """
    Epsilon-greedy policy
    """
    u = np.argmax(self._q(tt(x)).cpu().detach().numpy())
    r = np.random.uniform()
    if r < epsilon:
      return np.random.randint(self._action_dim)
    return u

  def train(self, episodes, time_steps, epsilon):

    stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))

    for e in range(episodes):
      print("%s/%s"%(e+1, episodes))
      s = env.reset()
      curr_action = sarsa.get_action(s, epsilon)
      for t in range(time_steps):
        ns, r, d, _ = env.step(curr_action)

        stats.episode_rewards[e] += r
        stats.episode_lengths[e] = t

        # Implement this
        # q_esti = self._q(tt(s).cpu().detach().numpy())
        q_esti = self._q(tt(s))
        q_target = q_esti.clone()

        self._q_optimizer.zero_grad()

        if (d):
            q_target[curr_action] = r
            loss_q_val = self._loss_function(q_esti, q_target)
            loss_q_val.backward()
            self._q_optimizer.step()
            break

        # get the Q values for next state: ns
        q_estim_next = self._q(tt(ns))
        next_action = self.get_action(ns, epsilon)
        q_target[curr_action] = r + self._gamma * q_estim_next[next_action]

        loss_q_val = self._loss_function(q_esti, q_target)
        loss_q_val.backward()
        self._q_optimizer.step()

        s = ns
        a = next_action
        # raise NotImplementedError("SARSA implementation is missing")

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
  env = MountainCarEnv()
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.n
  sarsa = SARSA(state_dim, action_dim, gamma=0.99)

  episodes = 1000
  time_steps = 200
  epsilon = 0.2

  stats = sarsa.train(episodes, time_steps, epsilon)

  plot_episode_stats(stats)

  for _ in range(5):
    s = env.reset()
    for _ in range(200):
      env.render()
      a = sarsa.get_action(s, epsilon)
      s, _, d, _ = env.step(a)
      if d:
        break
