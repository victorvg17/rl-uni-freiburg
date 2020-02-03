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
  return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)

class StateValueFunction(nn.Module):
  def __init__(self, state_dim, non_linearity=F.relu, hidden_dim=50):
    super(StateValueFunction, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, 1)
    self._non_linearity = non_linearity

  def forward(self, x):
    x = self._non_linearity(self.fc1(x))
    x = self._non_linearity(self.fc2(x))
    return self.fc3(x)

class Policy(nn.Module):
  def __init__(self, state_dim, action_dim, non_linearity=F.relu, hidden_dim=50):
    super(Policy, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, action_dim)
    self.m = nn.Softmax(dim=1)
    self._non_linearity = non_linearity

  def forward(self, x):
    x = self._non_linearity(self.fc1(x))
    x = self._non_linearity(self.fc2(x))
    return self.m(self.fc3(x))

class REINFORCE:
  def __init__(self, state_dim, action_dim, gamma):
    self._V = StateValueFunction(state_dim)
    self._pi = Policy(state_dim, action_dim)

    # self._V.cuda()
    # self._pi.cuda()

    self._gamma = gamma
    self._loss_function = nn.MSELoss()
    self._V_optimizer = optim.Adam(self._V.parameters(), lr=0.0001)
    self._pi_optimizer = optim.Adam(self._pi.parameters(), lr=0.0001)
    self._action_dim = action_dim
    self._loss_function = nn.MSELoss()

  def get_action(self, x):
    u = np.random.choice(np.arange(self._action_dim), p=self._pi(tt(np.array([x]))).cpu().detach().numpy()[0])
    return u

  def train(self, episodes, time_steps):
    stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))

    for i_episode in range(1, episodes + 1):
      # Generate an episode.
      # An episode is an array of (state, action, reward) tuples
      episode = []
      s = env.reset()
      for t in range(time_steps):
        a = self.get_action(s)
        ns, r, d, _ = env.step(a)

        stats.episode_rewards[i_episode-1] += r
        stats.episode_lengths[i_episode-1] = t

        episode.append((s, a, r))

        if d:
          break
        s = ns

      for t in range(len(episode)):
        # Find the first occurance of the state in the episode
        s, a, r = episode[t]
        # Sum up all rewards since the first occurance
        G = sum([e[2]*(self._gamma**i) for i,e in enumerate(episode[t:])])
        # Calculate average return for this state over all sampled episodes

        loss = self._loss_function(self._V(tt(np.array([s]))), tt(np.array([G])))
        self._V_optimizer.zero_grad()
        loss.backward()
        self._V_optimizer.step()

        action_per_policy = self._pi(tt(np.array([s])))
        log_action = torch.log(action_per_policy)
        td_error = G - self._V(tt(np.array([s])))
        # policy_objective = -(torch.log(self._pi(tt(np.array([s]))))[0, tt(np.array([a])).long()] * (G - self._V(tt(np.array([s]))))).mean()
        policy_objective = -(log_action[0, tt(np.array([a])).long()] * (td_error)).mean()
        # policy_objective = -
        self._pi_optimizer.zero_grad()
        policy_objective.backward()
        self._pi_optimizer.step()

      print("\r{} Steps in Episode {}/{}. Reward {}".format(len(episode), i_episode, episodes, sum([e[2] for i,e in enumerate(episode)])))
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
  reinforce = REINFORCE(state_dim, action_dim, gamma=0.99)

  episodes = 3000
  time_steps = 500

  stats = reinforce.train(episodes, time_steps)

  plot_episode_stats(stats)

  for _ in range(5):
    s = env.reset()
    for _ in range(500):
      env.render()
      a = reinforce.get_action(s)
      s, _, d, _ = env.step(a)
      if d:
        break
