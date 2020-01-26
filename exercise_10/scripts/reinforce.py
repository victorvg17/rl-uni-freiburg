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
  # return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
  return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)

class StateValueFunction(nn.Module):
  def __init__(self, state_dim, non_linearity=F.relu, hidden_dim=20):
    super(StateValueFunction, self).__init__()
    # Implement this!
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, 1)
    self._non_linearity = non_linearity
    # raise NotImplementedError("StateValueFunction.__init__ missing")

  def forward(self, x):
    # Implement this!
    x = self._non_linearity(self.fc1(x))
    x = self._non_linearity(self.fc2(x))
    # raise NotImplementedError("StateValueFunction.forward missing")
    return self.fc3(x)


class Policy(nn.Module):
  def __init__(self, state_dim, action_dim, non_linearity=F.relu, hidden_dim=20):
    super(Policy, self).__init__()
    # Implement this!
    self.fc1 = nn.Linear(in_features=state_dim, out_features=hidden_dim)
    self.fc2 = nn.Linear(in_features=hidden_dim, out_features=action_dim)
    # out_features for the last linear layer will be number of possible actions
    self.fc3 = nn.Softmax()
    self._non_linearity = non_linearity
    # raise NotImplementedError("Policy.__init__ missing")

  def forward(self, x):
    # Implement this!
    x = self._non_linearity(self.fc1(x))
    x = self._non_linearity(self.fc2(x))
    # raise NotImplementedError("Policy.forward missing")
    return self.fc3(x)


class REINFORCE:
  def __init__(self, state_dim, action_dim, gamma):
    self._V = StateValueFunction(state_dim)
    self._pi = Policy(state_dim, action_dim)

    # self._V.cuda()
    # self._pi.cuda()

    #adding alpha from last exercise.
    self._alpha = 0.001
    self._gamma = gamma
    self._loss_function = nn.MSELoss()
    self._V_optimizer = optim.Adam(self._V.parameters(), lr=0.0001)
    self._pi_optimizer = optim.Adam(self._pi.parameters(), lr=0.0001)
    self._action_dim = action_dim
    self._loss_function = nn.MSELoss()

  def get_action(self, state):
    # Implement this!
    # get action using self._pi
    state = tt(np.array([state]))
    action_prob = self._pi(state)
    highest_prob_action = np.random.choice(np.arange(0, self._action_dim), p=np.squeeze(action_prob.detach().numpy()))
    log_prob = torch.log(action_prob.squeeze(0)[highest_prob_action])
    return highest_prob_action, log_prob
    # raise NotImplementedError("REINFORCE.get_action missing")

  def _calc_value_function(self, episodes_data, curr_index):
      # total length of episodes
      G = 0
      T = len(episodes_data)
      assert T > curr_index, "curr_index should always be within maximum lenght of episodes"
      for k in range(curr_index+1, T):
          pow = k-(curr_index + 1)
          r_k = episodes_data[k][2]
          G = G + (self._gamma**pow)*r_k
      return G

  def _calc_policy_loss_gradient(self, input):
      input = torch.tensor([input], requires_grad=True)
      # input = Variable(torch.tensor([input]), requires_grad=True)
      self._pi_optimizer.zero_grad()
      input.backward()
      self._pi_optimizer.step()

  def _adjust_policy_parameters(self, itr):
    """
    # You can get the value of a parameter param by param.data and the gradient of param by param.grad.data.
    # You can overwrite the entry of a parameter param by param.data.copy_(new_value)
    """
    policy_params = self._pi.parameters()
    for param in policy_params:
        param_target = param.data + self._alpha*(self._gamma**itr)*param.grad.data
        param.data.copy_(param_target)


  def train(self, episodes, time_steps):
    stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))

    for i_episode in range(1, episodes + 1):
      # Generate an episode.
      # An episode is an array of (state, action, reward) tuples
      episode = []
      s = env.reset()
      for t in range(time_steps):
        highest_prob_action, log_prob = self.get_action(s)
        ns, r, d, _ = env.step(highest_prob_action)

        stats.episode_rewards[i_episode-1] += r
        stats.episode_lengths[i_episode-1] = t

        episode.append((s, highest_prob_action, r))

        if d:
          break
        s = ns

      for t in range(len(episode)):
        # Find the first occurance of the state in the episode
        s, a, r = episode[t]

        # calculate total value function
        G = self._calc_value_function(episodes_data=episode, curr_index=t)
        # call the function to calculate loss and gradient.
        policy_grad_ip = G*log_prob
        self._calc_policy_loss_gradient(input=policy_grad_ip)

        # update the parameters of policy function
        self._adjust_policy_parameters(itr=t)

        # Implement this!
        raise NotImplementedError("REINFORCE.train missing")

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
