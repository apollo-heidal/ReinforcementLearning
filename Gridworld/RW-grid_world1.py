'''
grid_world1.py
5x5 grid world from Chapter 3 Sutton and Barto
We will solve it using DP, so we will compute the expected reward for 
each action

'''

import numpy as np
import matplotlib.pyplot as plt
import random


class Grid_Cell:
    '''
    a grid cell will need to compute the rewards and new state for each action
    and return a reward based on them
    Each cell knows what actions are possible and the new states  associated with them
    '''
    N = 0
    E = 1
    W = 2
    S = 3
    global actions
    actions = [N, E, W, S]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.value = 0

    def list_rewards(self):
        'returns a list of tuples (reward, next state, probability) '
        ret_val = []
        for a in actions:
            # left border
            # bottom border
            # right border
            if self.x != 1 or self.y != 0:
                ret_val.append((0, 0.25))

        return ret_val

    def set_val(self, val):
        self.value = val

    def get_val(self):
        return self.value


class gridworld:
    def __init__(self, x=5, y=5): 
        self.grid = np.zeros((x, y)) 
        ranX = random.randrange(0, x) 
        ranY = random.randrange(0, y) 
        ranX2 = random.randrange(0, x) 
        ranY2 = random.randrange(0, y) 
        while (ranX == ranX2 and ranY == ranY2): 
            ranX2 = random.randrange(0, x) 
            ranY2 = random.randrange(0, y) 
            self.grid[ranX, ranY] = 10 
            self.grid[ranX2, ranY2] = 5 
            print(self.grid)

class Grid:
    '''
    2D array of Grid_Cells
    loop through all of the cells, update the values based on the expected reward and discounted value.
    v(s) = E(r + gamma * v(s'))
    the action a determines the reward r and the next state, but it can bex different for different states
    = sum over a of (r + gamma * v(s')) * p(a)

    '''

    def __init__(self, size=5, theta=0.01):
    
        self.epsilon = epsilon
        self.bandits = []
        # create k instances and append
        for i in range(k):
            q = np.random.randn()  # choose random q (Gaussian)
            b = Bandit(q)
            self.bandits.append(b)

    def simulate(self, runs, time):
        q_estimates = np.zeros(k)  # estimated values for each bandit
        counts = np.zeros(k)

        return q_estimates


if __name__ == '__main__':
    # 2000 runs for each epsilon
    # 1000 steps for each run
    epsilons = [0, 0.1, 0.01]
    # for eps in epsilons:
    #    print the q_estimates and average reward for each time step for each run
