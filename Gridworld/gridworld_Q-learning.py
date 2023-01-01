import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np


class Gridworld:
    def __init__(self, grid_size=10, random_start=False, animate=False, final_image=True):
        self.grid_size = grid_size
        self.random_start = random_start
        self.gamma = 0.99 # discount
        self.animate = animate
        self.final_image = final_image

        # [row][column][up,down,left,right]; +2 for grid border
        self.action_values = np.zeros((self.grid_size+2, self.grid_size+2, 4)) 

        # when the agent moves to A/B, the next move must be to A'/B' and will reward 10/5
        self.A = (1, 1) # top left of actual grid
        self.Ap = (self.A[0]+self.grid_size-2, self.A[1])
        self.B = (1, self.grid_size - 2) # -1 for indexing and -1 for border
        self.Bp = (self.B[0]+2, self.B[1])

        if self.animate:
            # 3D array with form [iteration, row, column] where value is the sum of all the action values (-1 dimension of policy)
            self.value_grid_iterations = np.sum(np.expand_dims(self.action_values, axis=0), axis=-1)
            self.agent_path = np.array([1,1]).reshape((1,2))


    def animationBuilder(self, r, c):
        # sum state/action values as "state_value_grid" 
        state_value_grid = np.sum(self.action_values, axis=-1)

        # add dimension to current state value grid to stack with previous value grids
        vg_reshaped = np.expand_dims(state_value_grid, axis=0)
        
        # stack all value grids) with current value grid
        self.value_grid_iterations = np.concatenate((self.value_grid_iterations, vg_reshaped))

        # complete path of agent over simulation; initialized in simulate()
        agent_location = np.array([r,c]).reshape((1,2))
        self.agent_path = np.concatenate((self.agent_path, agent_location))

    
    def showAnimation(self):
        #  main draw loop called by FuncAnimation class
        fig = plt.figure("Gridworld",figsize=(20,15))
        ax = fig.add_subplot(111)

        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        print(self.value_grid_iterations[-1])

        def animate(frame):
            r, c = self.agent_path[frame]
            grid_with_agent = self.value_grid_iterations[frame]
            grid_with_agent[r,c] = grid_with_agent.max()

            im = ax.matshow(grid_with_agent, cmap='coolwarm')
            cax.cla()
            fig.colorbar(im, cax=cax)
            return im,

        ani = FuncAnimation(fig, animate, frames=self.value_grid_iterations.shape[0], interval=10, blit=True)
        # ani.save("gw3-animation.mp4")
        plt.show()


    def showFinalValues(self):
        cell_values = self.action_values[1:-1,1:-1].sum(axis=-1)
        np.set_printoptions(precision=3, suppress=True)
        print(cell_values)

        ### cell values plot
        fig = plt.figure(figsize=(20,15))
        # remove start as outlier
        start_location = np.argmin(cell_values)

        cell_values.flat[start_location] = cell_values.max()
        cell_values.flat[start_location] = cell_values.min()
        im = plt.imshow(cell_values)
        plt.colorbar(im)
        plt.show()


        ### surface plot where zs are linearly spaced cell values
        value_index_pairs = [] # list of tuples (cell_value, cell_index)
        for i,v in enumerate(cell_values.flat):
            value_index_pairs.append((v,i))

        zs = np.empty(shape=cell_values.shape).flatten() # flat array to hold z values
        for i, (v,c) in enumerate(sorted(value_index_pairs)):
            # since the value_index_pairs list is sorted based on the cell value
            # i is a linearly spaced series that maps to the sorted cell_values array
            # c is used to recall the original index of value v and replace it with the linear value
            # this makes the small differences in cell values visible
            # but inaccurate since there are multiple collisions in the list of cell values
            zs[c] = len(value_index_pairs)-i # invert ordering so the agent "rolls downhill"
            # print(i, v, c)

        xs, ys = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        zs = zs.reshape(cell_values.shape)

        fig, ax = plt.subplots(figsize=(30,20), subplot_kw=dict(projection='3d'))
        ax.plot_surface(xs, ys, zs, cmap='plasma', shade=False)
        plt.show()


    def episode(self):
        if self.random_start:
            # start in random spot (not on border)
            r, c = np.random.randint(low=1, high=self.grid_size+1, size=2)
        else:
            r,c = self.grid_size - 1, self.grid_size // 2
        
        count = -1
        # explore the board for T steps
        while True:
            count += 1
            reward = 0

            if self.animate:
                self.animationBuilder(r, c)

            if (r,c) == self.Ap:
                # agent just moved to A' -> end episode
                break
            elif (r,c) == self.Bp:
                # '' ''         B'
                break


            ### core SARSA loop

            if (r,c) == self.A: # agent at A
                # now moves to A' with reward = 10
                sp_mva = self.action_values[self.Ap[0], self.Ap[1]].max()

                reward = 10
                ret = self.gamma**count * sp_mva
                self.action_values[r,c,:] += (ret + reward) / 4
                r,c = self.Ap
                continue
            elif (r,c) == self.B: # agent at B
                # now moves to B' with reward = 5
                reward = 5
                sp_mva = self.action_values[self.Bp[0], self.Bp[1]].max()

                ret = self.gamma**count * sp_mva
                self.action_values[r,c,:] += (ret + reward) / 4
                r,c = self.Bp
                continue
            else: # greedy choice
                # s' is the tuple coordinates of the mva
                a = self.getMVA(r, c)
                local_actions = self.getActions(r, c)
                sp = local_actions[a]

            # check to see if s' contains indices that place it in the border (0 or gridsize - 1)
            if self.outOfBounds(sp):
                # don't move but update the policy for that state/action pair
                self.action_values[r,c,a] -= 1
            else:
                # move to s' (not A,B,or out of bounds)
                reward = 0
                self.updateActionValues(r, c, a, sp, reward, count)
                r,c = sp

    
    def updateActionValues(self, r, c, a, sp, reward, count):
        # sp is tuple coordinates of the next state
        sp_mva = self.action_values[sp[0], sp[1]].max()
        ret = self.gamma**count * sp_mva
        self.action_values[r,c,a] += ret + reward


    def simulate(self, T):
        for e in range(T):
            self.episode()
            if e % 100 == 0:
                print(f"simulation is { (e / T) * 100 }% done ")
        
        if self.animate:
            self.showAnimation()
        
        if self.final_image:
            self.showFinalValues()    


    def getMVA(self, r, c):
        # get the values of available actions at (r,c)
        # nearby_state_values = self.estimateStateValues(actions)
        # array of booleans where the max value(s) become True and everything else is False
        # this allows ties to be broken randomly as opposed to always choosing the first occurence
        local_action_values = self.action_values[r,c]
        return np.random.choice( np.where(local_action_values == np.amax(local_action_values))[0] ) 


    def outOfBounds(self, sp):
        if np.isin(sp, [0,self.action_values.shape[0]-1]).any():
            # agent tried to move out of bounds
            return True
        else:
            return False


    def getActions(self, r, c):
        '''returns coordinates of cells surrounding (r,c): [up,down,left,right]'''
        return np.array([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
    

    def estimateStateValues(self, actions):
        # empty value estimates array for the actions; length of actions (usually 4)
        nearby_state_values = np.empty(actions.shape[0])

        for i,a in enumerate(actions):
            # row and column of this action
            r_a, c_a = a

            # sum action value estimates for each action of r,c (current cell)
            nearby_state_values[i] = np.sum(self.action_values[r_a,c_a])
        
        return nearby_state_values


if __name__ == "__main__":
    gw = Gridworld(animate=False)
    gw.simulate(T=1000)