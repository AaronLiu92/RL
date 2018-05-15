import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import *

Epsilon = 0.9 # greedy police, 90% choose the best action，10% choose random action
Alpha = 0.1 # learning rate
Lambda = 0.9 # Falloff reward value

class Agent:
    def __init__(self):
        self.currentpos = [0,0]

    def available_actions(self,position): # limit the movement of agent when he near the boundary
        Actions = []
        if position[0] > 0:
            Actions.append('left')
        if position[0] < 3:
            Actions.append('right')
        if position[1] > 0:
            Actions.append('down')
        if position[1] < 3:
            Actions.append('up')
        return Actions

    def feedback(self,picked_action):# based on the action (picked from available actions) to get next state postion and reward
        if picked_action == 'right':
            nextpos = [self.currentpos[0]+1,self.currentpos[1]]
        elif picked_action == 'up':
            nextpos = [self.currentpos[0], self.currentpos[1]+1]
        elif picked_action == 'left':
            nextpos = [self.currentpos[0]-1,self.currentpos[1]]
        else:
            nextpos = [self.currentpos[0], self.currentpos[1]-1]

        if (nextpos == [1,2]) or (nextpos == [3,1]) or (nextpos == [1,1]):
            R = -1
        elif nextpos == [3,2]:
            R = 1
        else:
            R = 0

        self.currentpos = nextpos # update position

        return nextpos, R

class Qlearning_table:
    def __init__(self):
        self.q_table = pd.DataFrame(columns=['left','right','up','down'],dtype=np.float64)
        self.q_table.index.name = 'state'

    def add_state(self,state):
        if state not in self.q_table.index:# add new state Series into pd.DataFrame(q_table)
            self.q_table = self.q_table.append(pd.Series([0]*4, index = self.q_table.columns, name = state))

    def choose_action(self, state, available_actions): # based on q_table value pick action(filtered by available actions)
        self.add_state(state)
        if (np.random.uniform() < Epsilon):
            # 选最好的action
            state_actions = self.q_table.loc[state,available_actions] #show left column:available_action, right column:their value
            #if all the actions have the same value, system auto-pick the left one, so we need mix the order of actions everytime
            state_actions = state_actions.reindex(np.random.permutation(state_actions.index))
            A = state_actions.idxmax()
        else:  # 10% choose random action
            A = np.random.choice(available_actions)
        return A

    def learn(self,state,action,next_state,reward):
        q_predict = self.q_table.loc[state,action]
        if next_state != [3,2]:
            q_target = reward + Lambda*self.q_table.loc[str(next_state),:].max()
        else:
            q_target = reward
        self.q_table.loc[state,action] += Alpha*(q_target-q_predict) #update q_table
        
class env:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('RL game1')
        self.window.geometry('200x200') # 4X4 unit length 50

        self.canvas = Canvas(master=self.window, bg='grey', width=5 * 50, height=5 * 50)

        for x0 in range(0, 200, 50):
            self.canvas.create_line(x0, 0, x0, 200, width=2)  # create vertical line on canvas
        for y0 in range(0, 200, 50):
            self.canvas.create_line(0, y0, 200, y0, width=2)  # create horizontal line on canvas

        agent_pos = [0,0] #these are all the coordinate of each project
        bad1 = [1,1]
        bad2 = [1,2]
        bad3 = [3,1]
        terminal = [3,2]

        self.bad1 = self.canvas.create_rectangle(bad1[0]*50 + 5, 155 - bad1[1]*50,
                                                 bad1[0]*50 + 45, 195 - bad1[1]*50, fill='black')
        #create_reactangle(left up x,y, right down x,y). calculate the corresponding coordinate in Canvas
        self.bad2 = self.canvas.create_rectangle(bad2[0]*50 + 5, 155 - bad2[1]*50,
                                                 bad2[0]*50 + 45, 195 - bad2[1]*50, fill='black')
        self.bad3 = self.canvas.create_rectangle(bad3[0]*50 + 5, 155 - bad3[1]*50,
                                                 bad3[0]*50 + 45, 195 - bad3[1]*50, fill='black')
        self.agent = self.canvas.create_rectangle(agent_pos[0]*50 + 5, 155 - agent_pos[1]*50,
                                                  agent_pos[0]*50 + 45, 195 - agent_pos[1]*50, fill='red')
        self.terminal = self.canvas.create_rectangle(terminal[0]*50 + 5, 155 - terminal[1]*50,
                                                     terminal[0]*50 + 45, 195 - terminal[1]*50, fill='yellow')

        self.canvas.pack()

    #each move,delete the agent project and re-create new agent with new coordinate
    def movement(self,location):
        self.canvas.delete(self.agent)
        self.agent = self.canvas.create_rectangle(location[0]*50 + 5, 155 - location[1]*50,
                                                  location[0]*50 + 45, 195 - location[1]*50, fill='red')

class RL:
    def __init__(self):
        q = Qlearning_table()
        a = Agent()
        grid = env()
        for round in range(35):
            a.currentpos = [0,0]
            state = str(a.currentpos)
            step_counter = 0

            grid.window.after(400,grid.movement([0,0]))
            grid.canvas.update()

            while not a.currentpos == [3,2]:
                available_choices = a.available_actions(a.currentpos)                        # find available actions
                action = q.choose_action(state, available_choices)                           # pick action based on available actions, and add state(index) into q_table
                nextpos, reward = a.feedback(action)                                         # get next state position and current state reward based on action

                grid.window.after(100,grid.movement(nextpos))                                # don't use time.sleep for canvas!!!!!!!!!!!
                grid.canvas.update()

                next_state = str(nextpos)                                                    # make index for next state
                q.add_state(next_state)                                                      # add next state ([0]*4 and index) into q_table

                q.learn(state, action, nextpos, reward)                                      # update q_table (Q learning)
                state = next_state                                                           # update current state
                step_counter += 1                                                            # count steps

            print("** There are ", step_counter, " Steps **")
        print(q.q_table)
        grid.window.mainloop() #in for-loop

if __name__ == '__main__':
    game = RL()
