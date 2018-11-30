import numpy as np
import time
import os
import math
import win_unicode_console

class Q_learning:
    def __init__(self,state_size,action_size,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.state_szie = state_size
        self.action_size = action_size
        self.Q_table = np.random.normal(0,0.3,(self.state_szie,self.action_size))
    
    def GetIndex(self,s):
        index_pow = np.arange(self.state_szie)
        index = np.sum(np.power(s*2,index_pow))-1
        return index
    
    def ChooseAction(self,s):
        index = self.GetIndex(s)
        a = np.argmax(self.Q_table[index])
        return a

    def Learning(self,s,s_,a,r):
        index = self.GetIndex(s)
        q_predict = self.Q_table[index,a]
        if s_ == None:
            q_target = r
        else:
            tmp = self.GetIndex(s_)
            q_target = r + np.amax(self.gamma*self.Q_table[tmp])
        self.Q_table[index,a] = lr * (q_target-q_predict)
    


        