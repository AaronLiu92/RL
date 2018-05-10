import numpy as np
import pandas as pd
import time

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

class RL:
    def __init__(self):
        q = Qlearning_table()
        a = Agent()
        for round in range(35):
            a.currentpos = [0,0]
            state = str(a.currentpos)
            step_counter = 0
            while not a.currentpos == [3,2]:
                available_choices = a.available_actions(a.currentpos)                        # find available actions
                action = q.choose_action(state, available_choices)                           # pick action based on available actions, and add state(index) into q_table
                nextpos, reward = a.feedback(action)#already change currentpos to nextpos    # get next state position and current state reward based on action
                next_state = str(nextpos)                                                    # make index for next state
                q.add_state(next_state)                                                      # add next state ([0]*4 and index) into q_table
                q.learn(state, action, nextpos, reward)                                      # update q_table (Q learning)
                state = next_state                                                           # update current state
                step_counter += 1                                                            # count steps
        print(q.q_table)
        print("-----------------------------------------------")
        print("** There are ", step_counter, " Steps **")


if __name__ == '__main__':
    game = RL()
