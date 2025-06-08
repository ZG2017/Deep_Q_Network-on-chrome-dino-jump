import numpy as np
import time
import pickle
import os
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import win_unicode_console
win_unicode_console.enable()

class DQNModel_v1(nn.Module):
    def __init__(self, states_size, action_size):
        super(DQNModel_v1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(states_size, 60),
            nn.ReLU(),
            nn.Linear(60, 15),
            nn.ReLU(),
            nn.Linear(15, action_size)
        )
    
    def forward(self, x):
        return self.net(x)
    
class DQNModel_v2(nn.Module):
    def __init__(self, states_size, action_size):
        """
        The DQN model with input dim being 2*states_size to handle speed up in later of the game.
        Args:
            states_size: int, the number of states
            action_size: int, the number of actions 
        """
        super(DQNModel_v2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(states_size*2, 60),
            nn.ReLU(),
            nn.Linear(60, 15),
            nn.ReLU(),
            nn.Linear(15, action_size)
        )
    
    def forward(self, x):
        return self.net(x)
    
class DQNModel_v3(nn.Module):
    def __init__(self, states_size, action_size):
        """
        The DQN model with input dim being 3*states_size to handle speed up in later of the game.
        Args:
            states_size: int, the number of states
            action_size: int, the number of actions 
        """
        super(DQNModel_v3, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(states_size*3, 60),
            nn.ReLU(),
            nn.Linear(60, 15),
            nn.ReLU(),
            nn.Linear(15, action_size)
        )
    
    def forward(self, state_t, state_t_plus_1):
        x = torch.cat([state_t, torch.abs(state_t_plus_1 - state_t), state_t_plus_1], dim=1)
        return self.net(x)

class DQN_Trainer:
    def __init__(
            self, 
            eval_net, 
            target_net, 
            lr, 
            epsilon_max, 
            epsilon_increase, 
            gamma, 
            number_of_states, 
            number_of_actions, 
            memory_size, 
            min_memory_count_to_start_training, 
            batch_size, 
            net_replace_memory_gap=100
        ):
        self.eval_net = eval_net
        self.target_net = target_net
        self.lr = lr
        self.epsilon_max = epsilon_max
        self.epsilon_init = 0
        self.epsilon_increase = epsilon_increase
        self.gamma = gamma
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.net_replace_memory_gap = net_replace_memory_gap
        self.memory_counter = 0
        self.last_replace_counter = min_memory_count_to_start_training  # Track the last memory counter when we replaced the network
        self.memory = np.zeros((self.memory_size, 2 * self.number_of_states + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def update_target_net(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def reset_trainer(self, init_epsilon, min_memory_count_to_start_training):
        self.memory_counter = 0
        self.last_replace_counter = min_memory_count_to_start_training
        self.epsilon_init = init_epsilon
        self.memory = np.zeros((self.memory_size, 2 * self.number_of_states + 2))

    def save_memory(self, s, s_, a, r):
        tmp = self.memory_counter % self.memory_size
        self.memory[tmp, :self.number_of_states] = s
        self.memory[tmp, self.number_of_states:2 * self.number_of_states] = s_
        self.memory[tmp, 2 * self.number_of_states] = a
        self.memory[tmp, 2 * self.number_of_states + 1] = r
        self.memory_counter += 1

    def choose_action(self, s):
        s = torch.FloatTensor(s).view(1, -1).to(self.device)
        with torch.no_grad():
            action_base = self.eval_net(s)
        tmp_a = torch.argmax(action_base)
        if np.random.uniform() < self.epsilon_init:
            a = tmp_a.item()
        else:
            a = np.random.randint(self.number_of_actions)
        return a

    def learning(self):
        # Check if we should update target network based on memory counter increment
        if self.memory_counter - self.last_replace_counter >= self.net_replace_memory_gap:
            self.update_target_net()
            self.last_replace_counter = self.memory_counter
            print("Target net has been replaced!")

        # train with lowest and highest reward memory
        # if self.memory_counter > self.memory_size:
        #     lowest_reward_memory_idx = np.argpartition(self.memory[:, 2 * self.number_of_states + 1], kth=self.batch_size//2)[:self.batch_size//2]
        #     highest_reward_memory_idx = np.argpartition(self.memory[:, 2 * self.number_of_states + 1], kth=self.memory_size-self.batch_size//2)[-self.batch_size//2:]
        # else:
        #     lowest_reward_memory_idx = np.argpartition(self.memory[:self.memory_counter, 2 * self.number_of_states + 1], kth=self.batch_size//2)[:self.batch_size//2]
        #     highest_reward_memory_idx = np.argpartition(self.memory[:self.memory_counter, 2 * self.number_of_states + 1], kth=self.memory_counter-self.batch_size//2)[-self.batch_size//2:]
        # batch_number = np.concatenate([lowest_reward_memory_idx, highest_reward_memory_idx])

        if self.memory_counter > self.memory_size:
            random_number = np.random.choice(self.memory_size, self.batch_size, replace=True)
            # max_reward = np.argpartition(self.memory[:, 2 * self.number_of_states + 1], kth=self.memory_size-self.batch_size//2)[-self.batch_size//2:]
        else:
            random_number = np.random.choice(self.memory_counter, self.batch_size, replace=True)
            # max_reward = np.argpartition(self.memory[:self.memory_counter, 2 * self.number_of_states + 1], kth=self.memory_counter-self.batch_size//2)[-self.batch_size//2:]
        # batch_number = np.concatenate([random_number, max_reward])
        batch_number = random_number
        batch_memory = self.memory[batch_number]
        b_s = torch.FloatTensor(batch_memory[:, :self.number_of_states]).to(self.device)
        b_s_ = torch.FloatTensor(batch_memory[:, self.number_of_states:2 * self.number_of_states]).to(self.device)
        b_a = batch_memory[:, 2 * self.number_of_states].astype(int)
        b_r = torch.FloatTensor(batch_memory[:, 2 * self.number_of_states + 1]).to(self.device)

        q_eval = self.eval_net(b_s)
        q_next = self.target_net(b_s_).detach()
        q_target = q_eval.clone().detach().cpu().numpy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, b_a] = b_r.cpu().numpy() + self.gamma * np.max(q_next.cpu().numpy(), axis=1)
        q_target = torch.FloatTensor(q_target).to(self.device)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon_init < self.epsilon_max:
            self.epsilon_init += self.epsilon_increase

    def load_memory(self, memory_file):
        self.memory = np.load(memory_file)
        self.memory_size = len(self.memory)

    def save_model(self, model_save_path):
        torch.save(self.eval_net.state_dict(), model_save_path)

    def load_model(self, file_path):
        self.eval_net.load_state_dict(torch.load(file_path, map_location=self.device))
        self.eval_net.eval()
        print(f"Model loaded: {file_path}")
