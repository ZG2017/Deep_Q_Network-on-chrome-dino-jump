import numpy as np
import time
import os
import math
import tensorflow as tf
import win_unicode_console
win_unicode_console.enable()

class DeepQNetwork():
    def __init__(self,lr,epsilon_max,epsilon_increase,gamma,number_of_states,number_of_actions,\
                 weight,replace_steps,memory_size,batch_size):  
        self.lr = lr
        self.epsilon_init = 0
        self.epsilon_max = epsilon_max
        self.epsilon_increase = epsilon_increase
        self.gamma = gamma
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.weight = weight
        self.replace_steps = replace_steps
        self.replace_counter = 0
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = np.zeros((self.memory_size,self.number_of_states*2+2))  # s,s_,r,a
        self.memory_counter = 0
        self.loss_his = []
        self.replace_his = []
        self.sess = tf.Session()

    
    def build_net(self):
        n_l1 = 100 # 60
        n_l2 = 25 # 15
        # eval net-------------------------------------------
        self.s = tf.placeholder(tf.float32,[None,self.number_of_states],name = "s")
        self.q_target = tf.placeholder(tf.float32,[None,self.number_of_actions],name = "a")
        with tf.variable_scope("eval_net"):
            c_names = ["eval_net_params",tf.GraphKeys.GLOBAL_VARIABLES]
            
            with tf.variable_scope("l1"):
                w1 = tf.get_variable("w1",[self.number_of_states,n_l1],initializer=tf.random_normal_initializer(0,0.3),\
                collections=c_names)
                b1 = tf.get_variable("b1",[1,n_l1],initializer=tf.constant_initializer(0.1),collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)

            with tf.variable_scope("l2"):
                w2 = tf.get_variable("w2",[n_l1,n_l2],initializer=tf.random_normal_initializer(0,0.3),collections=c_names)
                b2 = tf.get_variable("b2",[1,n_l2],initializer=tf.constant_initializer(0.1),collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1,w2)+b2)

            with tf.variable_scope("l3"):
                w3 = tf.get_variable("w3",[n_l2,self.number_of_actions],initializer=tf.random_normal_initializer(0,0.3),\
                collections=c_names)
                b3 = tf.get_variable("b3",[1,self.number_of_actions],initializer=tf.constant_initializer(0.1),collections=c_names)
                self.q_eval = tf.matmul(l2,w3)+b3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
                
        # target net-----------------------------------------
        self.s_ = tf.placeholder(tf.float32,[None,self.number_of_states],name = "s_")
        with tf.variable_scope("target_net"):
            c_names = ["target_net_params",tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope("l1"):
                w1 = tf.get_variable("w1",[self.number_of_states,n_l1],initializer=tf.random_normal_initializer(0,0.3),\
                collections=c_names)
                b1 = tf.get_variable("b1",[1,n_l1],initializer=tf.constant_initializer(0.1),collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_,w1)+b1)
            
            with tf.variable_scope("l2"):
                w2 = tf.get_variable("w2",[n_l1,n_l2],initializer=tf.random_normal_initializer(0,0.3),collections=c_names)
                b2 = tf.get_variable("b2",[1,n_l2],initializer=tf.constant_initializer(0.1),collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1,w2)+b2)
            
            with tf.variable_scope("l3"):
                w3 = tf.get_variable("w3",[n_l2,self.number_of_actions],initializer=tf.random_normal_initializer(0,0.3),\
                collections=c_names)
                b3 = tf.get_variable("b3",[1,self.number_of_actions],initializer=tf.constant_initializer(0.1),collections=c_names)
                self.q_next = tf.matmul(l2,w3)+b3

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    # save memoery
    def save_memory(self,s,s_,a,r):
        tmp = self.memory_counter%self.memory_size
        self.memory[tmp,:self.number_of_states] = s
        self.memory[tmp,self.number_of_states:2*self.number_of_states] = s_
        self.memory[tmp,2*self.number_of_states] = a
        self.memory[tmp,2*self.number_of_states+1] = r
        self.memory_counter += 1

    def choose_action(self,s):
        action_base = self.sess.run(self.q_eval,feed_dict={self.s:s})
        tmp_a = np.argmax(action_base)

        if np.random.uniform() < self.epsilon_init:
            a = tmp_a
        else:
            a = np.random.choice([0,1],1,p=self.weight)
            #a = np.random.randint(self.number_of_actions)
        return a

    def replace(self):
        t_params = tf.get_collection("target_net_params")
        e_params = tf.get_collection("eval_net_params")
        self.sess.run([tf.assign(t,e) for t,e in zip(t_params,e_params)])


    def learning(self):
        # see if needed to update target net
        if self.replace_counter%self.replace_steps == 0:
            self.replace()
            print("Target net has been replaced!")
            self.replace_his.append(0.1)
        else:
            self.replace_his.append(0)
        self.replace_counter += 1
        
        if self.memory_counter > self.memory_size:
            batch_number = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            batch_number = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[batch_number,:]

        q_eval,q_next = self.sess.run([self.q_eval,self.q_next],
        feed_dict = {self.s:batch_memory[:,:self.number_of_states],\
                     self.s_:batch_memory[:,self.number_of_states:2*self.number_of_states]})
        
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.number_of_states*2].astype(int)
        reward = batch_memory[:, self.number_of_states*2 + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        '''
        q_target[np.array(range(self.batch_size)),batch_memory[:,2*self.number_of_states].astype(int)] = \
        np.max(q_next,axis = 1)*self.gama + batch_memory[:,2*self.number_of_states+1]
        '''

        _,loss = self.sess.run([self.optimizer,self.loss],feed_dict={self.s:batch_memory[:,:self.number_of_states],
                                                                     self.q_target:q_target})

        self.loss_his.append(loss)

        if self.epsilon_init<self.epsilon_max:
            self.epsilon_init += self.epsilon_increase

    def plot(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        tmp = np.array(range(len(self.loss_his)))
        plt.plot(tmp,self.loss_his,"b-")  #,tmp,self.replace_his,"r."
        plt.show()

    def LoadMemoery(self,memoery_file):  # use LoadMemoery after check the format of data!!!
        self.memory = np.load(memoery_file)
        self.memory_size = self.memory.shape[0]

    def SaveModel(self,i,path):
        save_path = self.saver.save(self.sess, path, global_step=i)
        print(save_path)
    
    def LoadModel(self,file_path):
        self.saver.restore(self.sess,file_path)
        print("model loaded")


# example
"""
lr = 0.01
epsilon = 0.9
gamma = 0.9
max_epoch = 100

DQN_mou = DQN(lr = lr,
              epsilon_max = epsilon,
              gamma = gamma,
              number_of_states = env.observation_space.shape[0],
              number_of_actions= env.action_space.n,
              replace_steps = 100,
              memory_size = 2000,
              batch_size = 32)

DQN_mou.build_net()
"""