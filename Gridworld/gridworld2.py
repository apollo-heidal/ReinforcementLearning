from matplotlib.pyplot import colorbar, legend
import numpy as np
from numpy.lib.shape_base import expand_dims


class Gridworld:
    def __init__(self, grid_size, animate=True):
        self.grid_size = grid_size
        self.gamma = 0.9 # discount
        self.animate = animate

        # [row][column][up,down,left,right]; +2 for grid border
        self.policy_grid = np.zeros((self.grid_size+2, self.grid_size+2, 4)) 

        # when the agent moves to A/B, the next move must be to A'/B' and will reward 10/5
        self.A = (1, 2)
        self.B = (1, 4)

        if self.animate:
            self.value_grid_iterations = np.sum(np.expand_dims(self.policy_grid, axis=0), axis=-1)


    def animationBuilder(self, new_value_grid, r, c):
        # add dimension to stack with previous value grids
        vg_reshaped = np.expand_dims(new_value_grid, axis=0)
        agent_local = np.array([r,c]).reshape((1,2))
        
        # stack value grids
        self.value_grid_iterations = np.concatenate((self.value_grid_iterations, vg_reshaped))

        # stack agent locals
        self.agent_path = np.concatenate((self.agent_path, agent_local))

    
    def showAnimation(self):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        fig, ax = plt.subplots()

        def draw(frame):
            r, c = self.agent_path[frame][0], self.agent_path[frame][1]
            grid_with_agent = self.value_grid_iterations[frame]
            grid_with_agent[r][c] = 255

            im = plt.imshow(grid_with_agent, cmap='plasma')
            ax.text(0, 0, s=str(frame))
            return im,

        
        ani = FuncAnimation(fig, draw, frames=self.value_grid_iterations.shape[0], interval=10, blit=True)
        plt.show()


    def simulate(self, T):
        # start in random spot (not on border)
        r, c = np.random.randint(low=1, high=self.grid_size+1, size=2,)
        if self.animate:
            self.agent_path = np.array([r, c]).reshape((1, 2))
        
        # explore the board for T steps
        for t in range(T):
            if self.animate:
                value_grid = np.sum(self.policy_grid, axis=-1)
                self.animationBuilder(value_grid, r, c)

            actions = self.getActions(r,c)

            if np.random.rand() < 0.1:
                sp = actions[np.random.randint(actions.shape[0])]
            else:
                # get the values of available actions at (r,c)
                action_values = self.estimateActionValues(actions)

                # array of booleans where the max value(s) become True and everything else is False
                bool_max = (action_values == action_values.max())

                # choose most valuable action (break ties randomly)
                mva = np.argmax(bool_max * np.random.random(bool_max.shape))
            
                # s' is the tuple coordinates of the mva
                sp = actions[mva]

            # check to see if s' contains indices that place it in the border (0 or gridsize - 1)
            if self.outOfBounds(sp, cell_mva=(r,c,mva), action_values=action_values):
                continue

            # A found
            elif tuple(sp) == self.A:
                # print("agent found A")
                # update policy for r,c,move-to-A
                self.policy_grid[r,c,mva] += 0 + (self.gamma*np.sum(action_values))

                # move to A
                r,c = self.A
                t += 1

                # estimate action values for A'
                v_a = self.estimateActionValues(self.getActions(r,c))

                # update policy for A, move-to-A'
                self.policy_grid[r,c,:] += 10 / 4

                # move to A'
                r += self.grid_size-1
                
            # B found
            elif tuple(sp) == self.B:
                # print("agent found B")
                # update policy for r,c,move-to-B
                self.policy_grid[r,c,mva] += 0 + (self.gamma*np.sum(action_values))

                # move to B
                r,c = self.B
                t += 1

                # since B must go to B'
                # all the action value

                # estimate action values for B
                v_b = self.estimateActionValues(self.getActions(r,c))

                # update policy for B, move-to-B'
                self.policy_grid[r,c,:] += 5 / 4

                # move to B'
                r += 2
                
            else:
                # print("agent is exploring, legally")
                # update policy for r,c,move-to-mva (when not A/B)
                self.policy_grid[r,c,mva] += 0 + (self.gamma*np.sum(action_values))

                # move to s'
                r,c = sp


    def outOfBounds(self, sp, cell_mva: tuple, action_values):
        if np.isin(sp, [0,self.policy_grid.shape[0]-1]).any():
            # agent tried to move out of bounds
            # print("agent tried to move out of bounds")
            # reward -1 and don't move
            self.policy_grid[cell_mva] += -1 + (self.gamma*np.sum(action_values))
            return True
        else:
            return False


    def getActions(self, r, c) -> np.ndarray:
        # tuple coords of the cells surrounding (r,c): [up,down,left,right]
        return np.array([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
    

    def estimateActionValues(self, actions) -> np.ndarray:
        # empty value estimates array for the actions
        action_values = np.empty(actions.shape[0])

        for i,a in enumerate(actions):
            # row and column of this action
            r_a, c_a = a

            # sum action value estimates for each action of r,c (current cell)
            action_values[i] = np.sum(self.policy_grid[r_a,c_a])
        
        return action_values


if __name__ == "__main__":
    gs = 10
    gw = Gridworld(gs)
    gw.simulate(T=10000)
    gw.showAnimation()