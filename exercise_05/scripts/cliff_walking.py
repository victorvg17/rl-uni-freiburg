import numpy as np
import sys

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

def categorical_sample(prob_n, np_random):
  """
  Sample from categorical distribution
  Each row specifies class probabilities
  """
  prob_n = np.asarray(prob_n)
  csprob_n = np.cumsum(prob_n)
  return (csprob_n > np_random.rand()).argmax()

class CliffWalkingEnv:
  def _limit_coordinates(self, coord):
    coord[0] = min(coord[0], self.shape[0] - 1)
    coord[0] = max(coord[0], 0)
    coord[1] = min(coord[1], self.shape[1] - 1)
    coord[1] = max(coord[1], 0)
    return coord
  
  def _calculate_transition_prob(self, current, delta):
    new_position = np.array(current) + np.array(delta)
    new_position = self._limit_coordinates(new_position).astype(int)
    new_state = np.ravel_multi_index(tuple(new_position), self.shape)
    reward = -100.0 if self._cliff[tuple(new_position)] else -1.0
    is_done = self._cliff[tuple(new_position)] or (tuple(new_position) == (3,11))
    return [(1.0, new_state, reward, is_done)]

  def _seed(self, seed=None):
    #np.random.seed(seed)
    self.np_random = np.random.RandomState()
    self.np_random.seed(seed)
    return [seed]
  
  def __init__(self):
    self.shape = (4, 12)
    
    nS = np.prod(self.shape)
    nA = 4
    
    # Cliff Location
    self._cliff = np.zeros(self.shape, dtype=np.bool)
    self._cliff[3, 1:-1] = True
    
    # Calculate transition probabilities
    P = {}
    for s in range(nS):
      position = np.unravel_index(s, self.shape)
      P[s] = { a : [] for a in range(nA) }
      P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
      P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
      P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
      P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])
    
    # We always start in state (3, 0)
    isd = np.zeros(nS)
    isd[np.ravel_multi_index((3,0), self.shape)] = 1.0

    self._seed()
    self.isd = isd
    self.s = categorical_sample(self.isd, self.np_random)
    self.P = P
    self.nA = nA
    self.nS = nS
    self.lastaction = None

  def reset(self):
    self.s = categorical_sample(self.isd, self.np_random)
    self.lastaction = None
    return self.s

  def step(self, a):
    transitions = self.P[self.s][a]
    i = categorical_sample([t[0] for t in transitions], self.np_random)
    p, s, r, d= transitions[i]
    self.s = s
    self.lastaction = a
    return (s, r, d, {"prob" : p})