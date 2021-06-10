import numpy as np


class DPGridworld:
    def __init__(self) -> None:
        self.theta = 0.01  # threshold to break out of policy eval loop
        self.gridsize = 4

        # the value of each state under the current policy
        self.state_values = np.zeros(shape=(self.gridsize, self.gridsize))


    def evaluatePolicy(self):
        count=0
        while True:
            # for debug
            if count % 1 == 0:
                print(f"after {count} iterations")
                self.prettyPrintGrid()
            count+=1


            delta = 0
            # loop over all states
            for r in range(self.gridsize):
                for c in range(self.gridsize):
                    # skip if (r,c) is a terminal state
                    if (r,c) == (0,0) or (r,c) == (self.gridsize-1, self.gridsize-1):
                        continue
                    # else set reward
                    else:
                        reward = -1

                    # store old value for delta
                    old_v = self.state_values[r,c]

                    # gets a list of valid s' in tuple form: (r, c)
                    valid_moves = self.getValidMoves(r, c)

                    # get new state value by summing across valid actions
                    new_v = 0
                    for (sp_r, sp_c) in valid_moves:
                        sp_v = self.state_values[sp_r, sp_c]
                        new_v += (1/len(valid_moves)) * (reward + sp_v)

                    self.state_values[r, c] = new_v

                    # update delta
                    delta = abs(old_v-new_v)

            if delta < self.theta:
                print(f"delta({round(delta, 5)}) < theta({round(self.theta, 5)}) after {count} iterations")
                self.prettyPrintGrid()
                break


    def getValidMoves(self, r, c):
        moves = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
        valid_moves = []
        for a_r, a_c in moves:
            if a_r<0 or a_r>self.gridsize-1 or a_c<0 or a_c>self.gridsize-1:
                continue
            else:
                valid_moves.append((a_r, a_c))
        return valid_moves


    def prettyPrintGrid(self):
        for r in self.state_values:
            for v in r:
                print(round(v, 2), end="\t")
            print()
        print()


if __name__ == "__main__":
    gw = DPGridworld()
    gw.evaluatePolicy()