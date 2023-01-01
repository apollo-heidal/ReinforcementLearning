# programmed by Aubrey Birdwell
# reworked gridworld

import numpy as np
import matplotlib.pyplot as plt
import random

class Cell:
    def __init__(self, position = (0,0), stateValue = 0, count = 0, policy = 'E'):                       
        #self.action_choices = [(0,1), (0,-1), (1,0), (-1,0)]       
        self.position = position
        self.stateValue = stateValue
        self.count = count
        self.policy = policy
        self.setDirections()
        
    #set the action direction coordinates for the cell
    def setDirections(self):
        self.N = tuple(map(sum, zip((-1,0), self.position)))
        self.E = tuple(map(sum, zip((0,1), self.position)))
        self.S = tuple(map(sum, zip((1,0), self.position)))
        self.W = tuple(map(sum, zip((0,-1), self.position)))        

    # different access to the action coordinates
    # though still accessible by namespace.attribute
    def actionPosition(self, action):
        if action == 'N':
            return self.N
        if action == 'E':
            return self.E
        if action == 'S':
            return self.S
        if action == 'W':
            return self.W

    # simply returns all avail actions even if out of bounds?
    # maybe an improvment would be to make out of bounds actions unavail
    # then get rid of my out of bounds function in the gridworld class
    def actionsAvail(self):
        actions = []
        actions.append(self.N)
        actions.append(self.E)
        actions.append(self.S)
        actions.append(self.W)
        return actions

    # setter
    def setStateVal(self, value):
        self.stateValue = value
        
    #simple count function
    def updateCount(self):
        self.count = self.count + 1

    # setter
    def setPolicy(self, newPolicy):
        self.policy = newPolicy

    
class GridWorld:

    def __init__(self, theta = 0.1, gamma = 0.9):

        #only for grid world
        self.width = 5
        self.height = 5
        self.theta = theta
        self.gamma = 0.9

        #initialize the rows of cell objects
        gridworld = []
        for r in range(self.width):
            row_cells = []
            for c in range(self.height):
                grid_cell = Cell((r,c),0,0,'E')
                row_cells.append(grid_cell)
            gridworld.append(row_cells)
        self.grid = gridworld

    # count the out of bounds cells
    def numOutOfBounds(self,cell):
        actions = self.grid[cell[0]][cell[1]].actionsAvail()
        cnt = 0
        # check if in bounds
        for a in actions:
            if a[0] < 0 or a[0] > (self.width - 1):
                cnt += 1
            if a[1] < 0 or a[1] > (self.width - 1):
                cnt += 1                    
        return cnt

    # sum of all nbrs state values
    # add self if out of bounds 
    def getNbrsStates(self, cell):
        actions = self.grid[cell[0]][cell[1]].actionsAvail()
        state = self.grid[cell[0]][cell[1]].stateValue
        sumStates = 0
        # check if in bounds
        for a in actions:
            if a[0] < 0 or a[0] > (self.width - 1):
                sumStates += (1 * state)
            elif a[1] < 0 or a[1] > (self.width - 1):
                sumStates += (1 * state)
            else:
                sumStates += self.grid[a[0]][a[1]].stateValue
        return sumStates

    # rewards -1 out of bounds, 10 for (0,1) and 5 for (0,3), 0 all else
    def nbrsRewards(self, cell):
        if cell == (0,1):
            return 40
        if cell == (0,3):
            return 20
        return -1 * self.numOutOfBounds(cell)

    # returns nbrs state values
    def nbrsValues(self, cell):
        if cell == (0,1):
            return 4 * self.grid[4][1].stateValue
        if cell == (0,3):
            return 4 * self.grid[2][3].stateValue
        #otherwise
        return self.getNbrsStates(cell)

    # update cell instance state
    def updateCellState(self, cell):
        r = cell[0]
        c = cell[1]
        value = 0.25 * (self.nbrsRewards((r,c)) + (self.gamma * self.nbrsValues((r,c))))
        self.grid[r][c].setStateVal(value)

    # update cell policy
    def updateCellPolicy(self, cell):
        r = cell[0]
        c = cell[1]
        actions = self.grid[r][c].actionsAvail()
        values = []
        # check if in bounds
        for a in actions:
            if a[0] < 0 or a[0] > (self.width - 1):                
                values.append(-99)
            elif a[1] < 0 or a[1] > (self.width - 1):
                values.append(-99)
            else:
                values.append(self.grid[a[0]][a[1]].stateValue)
        maxval = max(values)
        maxpos = values.index(maxval)
        self.setPolicy(r, c, maxpos)

    # change 0-3 into N,E,S,W and access the setter in the cell class
    def setPolicy(self, r, c, p):
        if p == 0:
            policy = 'N'
        if p == 1:
            policy = 'E'
        if p == 2:
            policy = 'S'
        if p == 3:
            policy = 'W'
        self.grid[r][c].setPolicy(policy)

    # update all cells in rows and cols
    def updateGrid(self):
        for r in range(self.width):
            for c in range(self.height):
                self.updateCellState((r,c))
                self.updateCellPolicy((r,c))

    # print the updated grid...
    def showGridWorld(self):        
        for r in range(self.width):
            row = []
            for c in range(self.height):
                row.append((round(self.grid[r][c].stateValue, 1),self.grid[r][c].policy))
                #print(self.grid[r][c].position)
            print(row)


if __name__ == '__main__':

    cell1 = Cell()

    sim = GridWorld()
    for i in range(19):
        sim.updateGrid()

    sim.showGridWorld()
        
    
    


    
