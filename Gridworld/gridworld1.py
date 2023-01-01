import numpy as np
import matplotlib.pyplot as plt

'''
create grid
create grid values/probabilities; 4 for each cell (u,d,l,r)
in each cell:
    recursively check the value of neighbor-neighbor cells
    apply argmax and follow optimal path
    update values based on rewards/penalties received
'''
class Gridworld:
    def __init__(self, grid_size):
        self.g_s = grid_size
        self.gamma = 0.9

        self.policy_grid = np.zeros((self.g_s+2, self.g_s+2, 4)) # [row][column][up,down,left,right]; +2 for grid border

        # self.reward_grid = np.zeros((self.g_s+2, self.g_s+2))
        # # set border to -1
        # self.reward_grid[0] -= 1
        # self.reward_grid[-1] -= 1
        # self.reward_grid[1:-1,0] -= 1
        # self.reward_grid[1:-1,-1] -= 1

        # when the agent moves to A/B, the next move must be to A'/B' and will reward 10/5
        self.A = (1, 2)
        self.Ap = (-2, 2) # -2 row index means that even when the board size is increased the move from A to A' will be relatively large
        self.B = (1, 4)
        self.Bp = (3, 4)

        # print(self.reward_grid)

        # copies grid shape and adds dimension for up, down, left, and right, respectively
        # ptable_shape = (self.grid.shape[0],self.grid.shape[1],4) 
        # maps cardinal directions to indices of deepest axis, for readability
        # self.ptable_legend = {
        #     "N": 0,
        #     "E": 1,
        #     "S": 2,
        #     "W": 3 }

        # print(ptable_shape)
        # self.ptable = np.zeros(ptable_shape) # (row, col, [n,e,s,w])
        # self.tmp_ptable = np.copy(self.ptable)


    def simulate(self, runs, T):
        # start in random spot
        r, c = np.random.randint(1,self.g_s+1), np.random.randint(1,self.g_s+1)

        for t in range(T): # explore the board for T steps
            print(t, np.sum(self.policy_grid, axis=-1))

            actions = np.array([(r-1,c),(r+1,c),(r,c-1),(r,c+1)]) # possible actions from (r,c); [up,down,left,right]
            action_values = np.empty(actions.shape[0]) # value estimates for the actions
            for i,a in enumerate(actions):
                action_values[i] = self.estimateActionValue(a)
            
            # randomly choose index of most valuable action
            bool_max = (action_values == action_values.max())
            mva = np.argmax(bool_max * np.random.random(bool_max.shape))

            # s' is the tuple coordinates of the mva
            sp = actions[mva]
            
            # check to see if s' is outside of grid
            if sp[0]<1 or sp[0]>self.g_s or sp[1]<1 or sp[1]>self.g_s:
                # out of range; reward = -1 and no move
                self.policy_grid[r,c,mva] += -1 + (self.gamma*np.sum(action_values))
                continue

            elif tuple(sp) == self.A:
                # update policy for r,c,move-to-A
                self.policy_grid[r,c,mva] += 0 + (self.gamma*np.sum(action_values))

                # move to A
                r,c = self.A
                t += 1

                # estimate action values for A'
                v_ap = self.estimateActionValue(self.Ap)

                # update policy for A, move-to-A'
                self.policy_grid[r,c,:] += 10 + (self.gamma*4*v_ap)

                # move to A'
                r,c = self.Ap
                continue

            elif tuple(sp) == self.B:
                # update policy for r,c,move-to-B
                self.policy_grid[r,c,mva] += 0 + (self.gamma*np.sum(action_values))

                # move to B
                r,c = self.B
                t += 1

                # estimate action values for B'
                v_bp = self.estimateActionValue(self.Bp)

                # update policy for B, move-to-B'
                self.policy_grid[r,c,:] += 5 + (self.gamma*4*v_bp)

                # move to B'
                r,c = self.Bp
                continue

            else:
                # update policy for r,c,move-to-mva (when not A/B)
                self.policy_grid[r,c,mva] += 0 + (self.gamma*np.sum(action_values))

                # move to s'
                r,c = sp

    
    def estimateActionValue(self, cell):
        '''
        Takes cell tuple as input and returns the estimated value of moving to that cell based on the surround cells values.
        A/B have special conditions because the action taken from those cells has a higher reward. 
        '''
        r,c = cell
        return np.sum(self.policy_grid[r,c])


if __name__ == "__main__":
    gs = 5
    gw = Gridworld(gs)
    gw.simulate(1, 100)