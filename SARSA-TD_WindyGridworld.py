'''
hardcoded values match Example 6.5 in the book
'''
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

class WindyGridworld:
    def __init__(self) -> None:
        self.alpha = 0.1
        self.epsilon = 0.1

        # set start state
        self.start = (3,0)

        # set terminal state
        self.terminal_state = (3,7)

        # initialize state-action values; 7x10 grid with 4 actions
        self.action_values = np.zeros(shape=(7,10,4))

        # set wind values; applies "upward" to each column
        self.wind = np.array([0,0,0,1,1,1,2,2,1,0])

        self.frames = np.ones(shape=(1,7,10))

    
    def randomArgMax(self, arr):
        # returns the index of the highest value with ties broken randomly
        # _maxs is a boolean array where True values are occurrences of the max value
        max_as_bool = arr == arr.max()
        max_indices = np.flatnonzero(max_as_bool)
        return np.random.choice(max_indices)

    
    def takeAction(self, agent_pos, agent_action):
        # the action values are ordered such that 0-up, 1-right, 2-down, 3-left; like a clock
        r, c = agent_pos

        if agent_action == 0:
            r -= 1
        elif agent_action == 1:
            c += 1
        elif agent_action == 2:
            r += 1
        elif agent_action == 3:
            c -= 1
        
        # check whether agent tried to leave grid
        if r in [-1, self.action_values.shape[0]]:
            # agent tried to move off top/bottom, so return original pos
            return agent_pos
        elif c in [-1, self.action_values.shape[1]]:
            # same, but left/right
            return agent_pos
        else:
            # valid move
            return r,c

    
    def episode(self):
        # defines the steps that the agent takes from the starting point to the terminal state
        # set agent to start state
        agent_pos = self.start
        agent_path = [agent_pos]

        # choose a from s based on eps-greedy policy (max of action values for state s)
        if np.random.random() < self.epsilon:
            agent_action = np.random.randint(4)
        else:
            agent_action = self.randomArgMax(self.action_values[agent_pos])

        while True:
            # loop until agent finds terminal state
            if agent_pos == self.terminal_state:
                break
            
            # apply wind in agent's column
            gust = self.wind[agent_pos[1]]

            # init s'
            next_agent_pos = agent_pos
            for _ in range(gust):
                # move up as many times as the gust is strong
                next_agent_pos = self.takeAction(next_agent_pos, 0)

            # take action a with application of wind
            next_agent_pos = self.takeAction(next_agent_pos, agent_action)

            # store reward = -1 for all non-terminal moves
            if next_agent_pos == self.terminal_state:
                reward = 0
            else:
                reward = -1

            # choose a' from s' using greedy
            next_agent_action = self.randomArgMax(self.action_values[next_agent_pos])
            
            # q(s', a')
            q_sp_ap = self.action_values[next_agent_pos][next_agent_action]
            # q(s, a)
            q_s_a = self.action_values[agent_pos][agent_action]
            # update q(s,a) += alpha (reward + q(s',a') - q(s,a)); undiscounted, on-policy SARSA
            self.action_values[agent_pos][agent_action] += self.alpha * (reward + q_sp_ap - q_s_a)

            # s = s'
            agent_pos = next_agent_pos
            agent_path.append(agent_pos)

            # a = a'
            # on-policy means we only use the greedy action for our prediction: q(s', a')
            # but we include epsilon when actually making a move
            if np.random.random() < self.epsilon:
                agent_action = np.random.randint(4)
            else:
                agent_action = next_agent_action
        # return the agent's path on this episode
        return agent_path

    
    def simulate(self, n_episodes):
        # calls each episode and stores analysis data structures
        avg_steps = 0
        for e in range(n_episodes):
            if e % 1000 == 0:
                print(f"average steps after {e} episodes = {avg_steps}")
                self.printGreedyPolicy()
            agent_path = self.episode()
            if e % 1000 == 0:
                self.animationBuilder(agent_path)

            avg_steps += (len(agent_path)-avg_steps) / (e+1)


    def printGreedyPolicy(self):
        print(f"policy:")
        for r in self.action_values:
            for c in r:
                a = self.randomArgMax(c)
                if a == 0:
                    a = "^"
                elif a == 1:
                    a = ">"
                elif a == 2:
                    a = "v"
                elif a == 3:
                    a = "<"
                print(a, end=" ")
            print()
        print()


    def animationBuilder(self, agent_path):
        episode_frames = np.ones(shape=(len(agent_path),7,10))
        
        for f, pos in enumerate(agent_path):
            episode_frames[f][pos] = 0

        self.frames = np.concatenate((self.frames, episode_frames), axis=0)


    def showAnimation(self):
        #  main draw loop called by FuncAnimation class
        fig = plt.figure("Windy Gridworld",figsize=(20,15))
        ax = fig.add_subplot(111)

        # div = make_axes_locatable(ax)
        # cax = div.append_axes('right', '5%', '5%')
        # print(self.value_grid_iterations[-1])

        def animate(frame): 
            im = ax.matshow(self.frames[frame], cmap='plasma')
            # cax.cla()
            # fig.colorbar(im, cax=cax)
            return im,

        ani = FuncAnimation(fig, animate, frames=self.frames.shape[0], interval=10, blit=True)
        # ani.save("gw3-animation.mp4")
        plt.show()


if __name__ == "__main__":
    wgw = WindyGridworld()
    wgw.simulate(10000)
    wgw.showAnimation()
