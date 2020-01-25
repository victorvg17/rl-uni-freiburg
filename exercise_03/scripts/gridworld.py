import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv:
  """
  Grid World environment from Sutton's Reinforcement Learning book chapter 4.
  You are an agent on an MxN grid and your goal is to reach the terminal
  state at the top left or the bottom right corner.

  For example, a 4x4 grid looks as follows:

  T  o  o  o
  o  x  o  o
  o  o  o  o
  o  o  o  T

  x is your position and T are the two terminal states.

  You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
  Actions going off the edge leave you in your current state.
  You receive a reward of -1 at each step until you reach a terminal state.
  """

  def __init__(self, shape=[4,4]):
    if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
      raise ValueError('shape argument must be a list/tuple of length 2')

    self.shape = shape

    self.nS = np.prod(shape)
    self.nA = 4

    self.MAX_Y = shape[0]
    self.MAX_X = shape[1]

    P = {}
    grid = np.arange(self.nS).reshape(shape)
    it = np.nditer(grid, flags=['multi_index'])

    while not it.finished:
      s = it.iterindex  ## --x s = [0:15] x--
      y, x = it.multi_index

      P[s] = {a : [] for a in range(self.nA)}

      is_done = lambda s: s == 0 or s == (self.nS - 1)
      reward = 0.0 if is_done(s) else -1.0

      if is_done(s):
        P[s][UP] = [(1.0, s, reward, True)]
        P[s][RIGHT] = [(1.0, s, reward, True)]
        P[s][DOWN] = [(1.0, s, reward, True)]
        P[s][LEFT] = [(1.0, s, reward, True)]
      else:
        ns_up = s if y == 0 else s - self.MAX_X
        ns_right = s if x == (self.MAX_X - 1) else s + 1
        ns_down = s if y == (self.MAX_Y - 1) else s + self.MAX_X
        ns_left = s if x == 0 else s - 1
        P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
        P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
        P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
        P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

      it.iternext()

    self.P = P
