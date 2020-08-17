#!/usr/bin/env python
# coding: utf-8

# # ML Task - 01 (The Robotics Forum)

# ### Mentor Shripad Kulkarni
# #### Preeti Oswal, Aryan Gupta, Avinash Vijayvargiya
# 

# ## Importing all required lib

# In[1]:


import numpy as np
import pickle


# ### Developing board and funcation to game play

# In[2]:


class Game:
    def __init__(self, p1, p2):    
        self.board = np.zeros((3, 3))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = 1
    
    
    def getHash(self):
        self.boardHash = str(self.board.reshape(9))
        return self.boardHash
    
    
    ## DECLARING CONDITION TO WIN THE GAME
    def winner(self):
        
        ## CONDITION FOR ROWS
        for i in range(3):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
            
        ## CONDITION FOR COLUMNS
        for i in range(3):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
            
        ## CONDITION FOR DIAGONALS
        sum1_diag = sum([self.board[i, i] for i in range(3)])
        sum2_diag = sum([self.board[i, 3-i-1] for i in range(3)])
        if sum1_diag == 3 or sum2_diag == 3:
            self.isEnd = True
            return 1
        if sum1_diag == -3 or sum2_diag == -3:
            self.isEnd = True
            return -1
        if len(self.remainingPositions()) == 0:
            self.isEnd = True
            return 0
        self.isEnd = False
        return None
    
    ## FUNCTION TO DETERMINE VACANT POSITIONS
    def remainingPositions(self):
        positions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        return positions
    
    ## FUNCTION TO UPDATE BOARD VALUES
    def update(self, position):
        self.board[position] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1
    
    
    ##  FUNCTION TO REWARD MACHINE
    def giveReward(self):
        result = self.winner()
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)
    
    ## FUNCTION TO RESET THE BOARD FOR THE GAME
    def reset(self):
        self.board = np.zeros((3,3))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1
        
    ## FUNCTION FOR PLAYER 1 AND PLAYER 2
    def play(self, rounds=100):
        for i in range(rounds):
            while not self.isEnd:
                positions = self.remainingPositions()
                p1_action = self.p1.NextMove(positions, self.board, self.playerSymbol)
                self.update(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    positions = self.remainingPositions()
                    p2_action = self.p2.NextMove(positions, self.board, self.playerSymbol)
                    self.update(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)
                    
                    win = self.winner()
                    if win is not None:
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
    
    def play2(self):
        while not self.isEnd:
            positions = self.remainingPositions()
            p1_action = self.p1.NextMove(positions, self.board, self.playerSymbol)
            self.update(p1_action)
            self.Display()
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "Wins! , Better Luck next time")
                else:
                    print("Tie!")
                self.reset()
                break

            else:
                positions = self.remainingPositions()
                p2_action = self.p2.NextMove(positions)
                self.update(p2_action)
                self.Display()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "Win!")
                    else:
                        print("Tie!")
                    self.reset()
                    break

    def Display(self):
        for i in range(0, 3):
            print('-------------')
            out = '| '
            for j in range(0, 3):
                if self.board[i, j] == 1:
                    token = 'X'
                if self.board[i, j] == -1:
                    token = 'O'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')


# ### Creating self-learning machine model

# In[3]:


class Model:
    def __init__(self, name, random=0.3):
        self.name = name
        self.states = []
        self.random = random
        self.lr = 0.2
        self.states_value = {}
    
    def getHash(self, board):
        boardHash = str(board.reshape(9))
        return boardHash
    
    def NextMove(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.random:
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                if value >= value_max:
                    value_max = value
                    action = p
        return action
    
    def addState(self, state):
        self.states.append(state)
    
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr*(0.9*reward - self.states_value[st])
            reward = self.states_value[st]
            
    def reset(self):
        self.states = []
        
    def saveDict(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadDict(self, file):
        fr = open(file,'rb')
        self.states_value = pickle.load(fr)
        fr.close()


# ### When Human is player

# In[4]:


class Human:
    def __init__(self, name):
        self.name = name 
    def NextMove(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action
    def addState(self, state):
        pass
    def feedReward(self, reward):
        pass      
    def reset(self):
        pass


# ### Training the model

# In[5]:


p1 = Model("p1")
p2 = Model("p2")

st = Game(p1, p2)
print("training...")
st.play(50000)


# In[6]:


p1.saveDict()
p2.saveDict()


# In[7]:


p1.loadDict("policy_p1")


# In[8]:


p1 = Model("computer" , 0)
p1.loadDict("policy_p1")

p2 = Human("You")
st = Game(p1, p2)
st.play2()

