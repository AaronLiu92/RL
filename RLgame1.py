import tkinter as tk
import numpy as np
import pandas as pd
import time
from tkinter import *

actions = ['left','right','up','down']

grid_width = 5
grid_height = 5
UNIT = 50

alpha = 0.01 # Learning rate
epsion = 0.9  # greedy policy, random = 0-90% choose best action, 0-10% choose random
gamma = 0.9 # attenuation for future R

class Agent:
    def __init__ (self, current_pos=[25,225]):
        self.current_pos = current_pos

    def action_pick(self,state_num, table):# Q values is the value list of q table(certain state)
        if np.random.rand() < epsion:
            state_values = table.loc[state_num,:] #show all actions q values of current state
            pick = state_values.idxmax() # show action with max q value
        else:
            pick = np.random.choice(actions)
        return pick

    def move_it(self, action):
        if (action == 'left') and (self.current_pos[0] > 25):
            move_value = [-50,0]
        elif (action == 'right') and (self.current_pos[0] < 225):
            move_value = [50, 0]
        elif (action == 'up') and (self.current_pos[1] > 25):
            move_value = [0, -50]
        elif (action == 'down') and (self.current_pos[1] < 225):
            move_value = [0, 50]
        else:
            move_value=[0,0]

        self.next_pos = [self.current_pos[0] + move_value[0],self.current_pos[1]+move_value[1]]
        self.current_pos = self.next_pos

        if self.next_pos == [175,75]:
            reward = +1
            done = True

        elif (self.next_pos == [75,125]) or (self.next_pos == [125,75]) or \
            (self.next_pos == [175, 175]):
            reward = -1
            done = False

        else:
            reward = 0
            done = False

        return self.next_pos, reward, done

class Qtable:
    def __init__ (self):
        self.qtable = pd.DataFrame(columns=actions, dtype=np.float64)

    def step(self):
        self.qtable = self.qtable.append(pd.Series([0]*4,index=actions,name='0'))

    def learn(self):
        if self.next_pos != [175,75]:
            q_target =  reward + gamma*self.qtable.loc[self.next_pos,:].max()
class Env:
    def __init__ (self):
        self.window = tk.Tk()
        self.window.title('RL game1')
        self.window.geometry('250x250')
        self.create_grid()

    def create_grid(self):

        self.canvas = Canvas(master=self.window, bg='grey', width=5 * 50, height=5 * 50)

        for x0 in range(0,250,50):
            self.canvas.create_line(x0,0,x0,250,width=2) #create vertical line on canvas
        for y0 in range(0,250,50):
            self.canvas.create_line(0,y0,250,y0,width=2) #create horizontal line on canvas

        start_center = np.array([25, 225]) # create the center of the starting grid
        good_center = start_center + np.array([3*UNIT, -3*UNIT]) # create the center of good grid(3 grids right&up of starting grid
        bad1_center = start_center + np.array([UNIT, -2*UNIT])
        bad2_center = start_center + np.array([2*UNIT, -3*UNIT])
        bad3_center = start_center + np.array([3 * UNIT, -1 * UNIT])

        # size of the grid is 50 x 50, so the radio of the grid is 25
        # bad1, bad2 and bad 3are the location with -1 Reward
        # good point has +1 Reward
        self.bad1 = self.canvas.create_rectangle(bad1_center[0]-20,bad1_center[1]-20,
                                            bad1_center[0]+20,bad1_center[1]+20,fill='black')

        self.bad2 = self.canvas.create_rectangle(bad2_center[0] - 20, bad2_center[1] - 20,
                                            bad2_center[0] + 20, bad2_center[1] + 20, fill='black')

        self.bad3 = self.canvas.create_rectangle(bad3_center[0] - 20, bad3_center[1] - 20,
                                            bad3_center[0] + 20, bad3_center[1] + 20, fill='black')

        self.good = self.canvas.create_oval(good_center[0]-20,good_center[1]-20,
                                       good_center[0]+20,good_center[1]+20,fill='yellow')

        self.people = self.canvas.create_rectangle(start_center[0]-20,start_center[1]-20,
                                                start_center[0]+20,start_center[1]+20,fill='red')

        self.canvas.pack()
        print(self.canvas.coords(self.people))

class RL:
    def __init__ (self):
        pass

Q = Qtable()
print('Qtable: ',Q.qtable)
a = Agent()

Q.step()
print(Q.qtable)
print(a.move_it('right')) #reward[1]

