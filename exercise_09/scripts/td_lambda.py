import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gridworld import GridworldEnv
import numpy as np

optimal_policy = [0, 3, 3, 2, 0, 0, 0, 2, 0, 0, 1, 2, 0, 1, 1, 0]

def get_action(s):
  return optimal_policy[s]

def tt(ndarray):
  # Untoggle to use CUDA
  # return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
  return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)

class Network(nn.Module):
  def __init__(self, state_dim, non_linearity=F.relu, hidden_dim=10):
    super(Network, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, 1)
    self._non_linearity = non_linearity

  def forward(self, x):
    x = self._non_linearity(self.fc1(x))
    x = self._non_linearity(self.fc2(x))
    return self.fc3(x)

  def init_zero(self):
    torch.nn.init.constant_(self.fc1.weight, 0)
    torch.nn.init.constant_(self.fc1.bias, 0)
    torch.nn.init.constant_(self.fc2.weight, 0)
    torch.nn.init.constant_(self.fc2.bias, 0)
    torch.nn.init.constant_(self.fc3.weight, 0)
    torch.nn.init.constant_(self.fc3.bias, 0)

def overwrite_params(weights_old, weights_updated):
    for weight_old, weight_updated in zip(weights_old, weights_updated):
        weight_old.data.copy_(weight_updated)



class TDLambda:
  def __init__(self, state_dim, gamma, trace_decay, alpha):
    self.v = Network(state_dim)
    self.z = Network(state_dim)

    # Untoggle to use CUDA
    # self.v.cuda()
    # self.z.cuda()
    self.gamma = gamma
    self.trace_decay = trace_decay
    self.alpha = alpha

    self.z.init_zero()

  def train(self, episodes, time_steps):
    for e in range(episodes):
      if (e+1)%100 == 0:
        print("%s/%s"%(e+1, episodes))
      s = env.reset()

      # Implement this
      # Initialize z each episode
      """
          z: ET exist for each state and determines how much of respective component of
          weight vector (dimension = s*1) should be used for updating weight for next
          iteration
      """
      self.z.init_zero()

      for t in range(time_steps):
        a = get_action(s)
        ns, r, d = env.step(a)

        # Implement this
        # self.v(torch.tensor(s)).mean().backward()
        # calculate TD error
        value_fun_current = self.v.forward(tt(np.array(s)))

        z_params = self.z.parameters()
        v_params = self.v.parameters()
        # model(torch.randn(1, 3, 224, 224)).mean().backward()

        for z_param, v_param in zip(z_params, v_params):
            z_param_new_val = self.trace_decay*self.gamma*z_param.data + v_param.grad
            z_param.data.copy_(z_param_new_val)
        # calculate TD error

        value_fun_next = self.v(ns)
        td_errors = r + self.gamma*value_fun_next - value_fun_current

        """
        # You can get the value of a parameter param by param.data and the gradient of param by param.grad.data.
        # You can overwrite the entry of a parameter param by param.data.copy_(new_value)
        """
        v_params = self.v.parameters()
        for v_param, td_error, z_param in zip(v_params, td_errors, z_params):
            v_param_new_val = v_param.data + self.alpha*td_error*z_param
            v_param.data.copy_(v_param_new_val)

        weights_old = self.v.parameters()
        overwrite_params(weights_old = weights_old, weights_updated = v_params)

        if d:
          break

        s = ns

    return self.v

if __name__ == "__main__":
  env = GridworldEnv()
  tdlambda = TDLambda(1, gamma=0.99, trace_decay=0.5, alpha=0.001)

  episodes = 1000000
  time_steps = 50

  w = tdlambda.train(episodes, time_steps)

  for i in range(env.nS):
    print("%s: %s" %(i, w(tt(np.array([i])))))
