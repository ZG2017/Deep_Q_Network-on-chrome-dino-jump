"""
Actor-Critic Implementation using TD Learning

This module implements an Actor-Critic algorithm that combines:
- Policy network (Actor) for action selection
- Value network (Critic) for state value estimation using TD learning
- Replay memory system for experience storage and batch training

The value network helps reduce variance in policy gradient estimates by providing
baseline estimates for advantage calculation. Uses Temporal Difference (TD) learning
instead of Monte Carlo returns for more efficient and stable learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        The Actor network (policy network) with input dim being 3*states_size to handle speed up in later of the game.
        Args:
            states_size: int, the number of states
            action_size: int, the number of actions 
        """
        super(ActorNetwork, self).__init__()
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_size*3, 120),
            nn.ReLU(),
            nn.Linear(120, 30),
            nn.ReLU(),
            nn.Linear(30, action_size),
        )
                
    def forward(self, state_t, state_t_plus_1):
        # Policy network
        x = torch.cat([state_t, torch.abs(state_t_plus_1 - state_t), state_t_plus_1], dim=1)
        action_probs = F.softmax(self.policy_net(x), dim=-1)
        return action_probs

class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        """
        Critic network to estimate state values for Actor-Critic architecture.
        Args:
            state_size: int, the number of states
        """
        super(CriticNetwork, self).__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(state_size*3, 120),
            nn.ReLU(),
            nn.Linear(120, 30),
            nn.ReLU(),
            nn.Linear(30, 1),  # Output single value estimate
        )
                
    def forward(self, state_t, state_t_plus_1):
        x = torch.cat([state_t, torch.abs(state_t_plus_1 - state_t), state_t_plus_1], dim=1)
        value = self.value_net(x)
        return value


class ActorNetwork_v2(nn.Module):
    def __init__(self, state_size, action_size):
        """
        The Actor network (policy network) with input dim being 3*states_size to handle speed up in later of the game.
        Args:
            states_size: int, the number of states
            action_size: int, the number of actions 
        """
        super(ActorNetwork_v2, self).__init__()
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_size*3, 20),
            nn.ReLU(),
            nn.Linear(20, action_size),
        )
                
    def forward(self, state_t, state_t_plus_1):
        # Policy network
        x = torch.cat([state_t, torch.abs(state_t_plus_1 - state_t), state_t_plus_1], dim=1)
        action_probs = F.softmax(self.policy_net(x), dim=-1)
        return action_probs

class CriticNetwork_v2(nn.Module):
    def __init__(self, state_size):
        """
        Critic network to estimate state values for Actor-Critic architecture.
        Args:
            state_size: int, the number of states
        """
        super(CriticNetwork_v2, self).__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(state_size*3, 20),
            nn.ReLU(),
            nn.Linear(20, 1),  # Output single value estimate
        )
                
    def forward(self, state_t, state_t_plus_1):
        x = torch.cat([state_t, torch.abs(state_t_plus_1 - state_t), state_t_plus_1], dim=1)
        value = self.value_net(x)
        return value

class ActorCriticTrainer:
    def __init__(self, actor_net, critic_net, lr=0.001, critic_lr=0.001, gamma=0.99, memory_size=10000, batch_size=32, min_memory_count_to_start_training=1000):
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.actor_optimizer = optim.AdamW(actor_net.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(critic_net.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.min_memory_count_to_start_training = min_memory_count_to_start_training
        
        # Replay memory system (similar to DQN) - now includes done flags for TD learning
        self.memory_counter = 0
        self.number_of_states = actor_net.policy_net[0].in_features // 3  # Extract state size from network
        self.memory = np.zeros((self.memory_size, 4 * self.number_of_states + 3))  # +3 for action, reward, done
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def save_memory(self, s_1, s_2, s_1_, s_2_, a, r, done=0):
        """Save experience to replay memory with done flag for TD learning"""
        tmp = self.memory_counter % self.memory_size
        self.memory[tmp, 0 * self.number_of_states:1 * self.number_of_states] = s_1
        self.memory[tmp, 1 * self.number_of_states:2 * self.number_of_states] = s_2
        self.memory[tmp, 2 * self.number_of_states:3 * self.number_of_states] = s_1_
        self.memory[tmp, 3 * self.number_of_states:4 * self.number_of_states] = s_2_
        self.memory[tmp, 4 * self.number_of_states] = a
        self.memory[tmp, 4 * self.number_of_states + 1] = r
        self.memory[tmp, 4 * self.number_of_states + 2] = done
        self.memory_counter += 1
    
    def load_memory(self, memory_file):
        """Load memory from file (similar to DQN)"""
        self.memory = np.load(memory_file)
        self.memory_size = len(self.memory)
        
    def reset_trainer(self, memory_size):
        """Reset the trainer with new memory size"""
        self.memory_counter = 0
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, 4 * self.number_of_states + 3))
        
    def choose_action(self, s_1, s_2):
        s_1 = torch.FloatTensor(s_1).view(1, -1).to(self.device)
        s_2 = torch.FloatTensor(s_2).view(1, -1).to(self.device)
        with torch.no_grad():
            action_probs = self.actor_net(s_1, s_2)
        # Sample action from policy
        action_probs_cpu = action_probs.cpu().numpy()[0]
        action = np.random.choice(len(action_probs_cpu), p=action_probs_cpu)
        return action
    
    def choose_action_deterministic(self, s_1, s_2):
        """Choose action using the actor network"""
        s_1 = torch.FloatTensor(s_1).view(1, -1).to(self.device)
        s_2 = torch.FloatTensor(s_2).view(1, -1).to(self.device)
        with torch.no_grad():
            action_probs = self.actor_net(s_1, s_2)
        action_probs_cpu = action_probs.cpu().numpy()[0]
        action = np.argmax(action_probs_cpu)
        return action
    
    def get_value_estimate(self, s_1, s_2):
        """Get value estimate for given state"""
        s_1 = torch.FloatTensor(s_1).view(1, -1).to(self.device)
        s_2 = torch.FloatTensor(s_2).view(1, -1).to(self.device)
        with torch.no_grad():
            value = self.critic_net(s_1, s_2)
        return value.item()
    
    def get_advantage_estimate(self, s_1, s_2, returns):
        """Get advantage estimate for given state and returns"""
        s_1 = torch.FloatTensor(s_1).view(1, -1).to(self.device)
        s_2 = torch.FloatTensor(s_2).view(1, -1).to(self.device)
        with torch.no_grad():
            value = self.critic_net(s_1, s_2)
        advantage = returns - value.item()
        return advantage
    
    def store_experience(self, state, state_plus_1, action, reward, next_state=None, next_state_plus_1=None, done=0):
        """Store experience for training using replay memory with done flag"""
        if next_state is None:
            next_state = state
        if next_state_plus_1 is None:
            next_state_plus_1 = state_plus_1
        self.save_memory(state, state_plus_1, next_state, next_state_plus_1, action, reward, done)
    
    def compute_td_targets(self, rewards, next_state_values, gamma, dones=None):
        """Compute TD targets using next state values"""
        if dones is None:
            dones = np.zeros_like(rewards)  # Assume all episodes continue if not provided
        
        td_targets = []
        for i in range(len(rewards)):
            if dones[i]:  # Terminal state
                td_target = rewards[i]
            else:  # Non-terminal state
                td_target = rewards[i] + gamma * next_state_values[i]
            td_targets.append(td_target)
        
        return torch.FloatTensor(td_targets).to(self.device)
    
    def train(self):
        """Train both policy and value networks using Actor-Critic with replay memory"""
        # Check if we have enough experiences to start training
        if self.memory_counter < self.min_memory_count_to_start_training:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        # Sample batch from replay memory
        if self.memory_counter > self.memory_size:
            batch_indices = np.random.choice(self.memory_size, self.batch_size, replace=True)
        else:
            batch_indices = np.random.choice(self.memory_counter, self.batch_size, replace=True)
        
        batch_memory = self.memory[batch_indices]
        
        # Extract batch data
        b_s_1 = torch.FloatTensor(batch_memory[:, 0 * self.number_of_states:1 * self.number_of_states]).to(self.device)
        b_s_2 = torch.FloatTensor(batch_memory[:, 1 * self.number_of_states:2 * self.number_of_states]).to(self.device)
        b_s_1_ = torch.FloatTensor(batch_memory[:, 2 * self.number_of_states:3 * self.number_of_states]).to(self.device)
        b_s_2_ = torch.FloatTensor(batch_memory[:, 3 * self.number_of_states:4 * self.number_of_states]).to(self.device)
        b_a = batch_memory[:, 4 * self.number_of_states].astype(int)
        b_r = batch_memory[:, 4 * self.number_of_states + 1]
        b_done = batch_memory[:, 4 * self.number_of_states + 2].astype(int)
        
        # Forward pass for both networks
        action_probs = self.actor_net(b_s_1, b_s_2)
        state_values = self.critic_net(b_s_1, b_s_2).squeeze()
        
        # Compute next state values for TD targets
        with torch.no_grad():
            next_state_values = self.critic_net(b_s_1_, b_s_2_).squeeze()
        
        # Compute TD targets (r + γ * V(s')) using done flags
        td_targets = self.compute_td_targets(b_r, next_state_values.cpu().numpy(), self.gamma, b_done)
        
        # Compute advantages (TD targets - value estimates)
        advantages = td_targets - state_values.detach()
        
        # Policy loss using advantages (Actor-Critic)
        log_action_probs = torch.log(action_probs + 1e-8)
        selected_log_action_probs = log_action_probs.gather(1, torch.LongTensor(b_a).unsqueeze(1).to(self.device)).squeeze(1)
        policy_loss = -(selected_log_action_probs * advantages).mean()
        
        # Value loss (MSE between value estimates and TD targets)
        value_loss = F.mse_loss(state_values, td_targets)
        
        # Backward pass for actor network
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Backward pass for critic network
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def save_model(self, actor_filepath, critic_filepath=None):
        """Save both actor and critic models"""
        torch.save(self.actor_net.state_dict(), actor_filepath)
        if critic_filepath is not None:
            torch.save(self.critic_net.state_dict(), critic_filepath)
        else:
            # Save critic network with similar name as actor
            critic_filepath = actor_filepath.replace('.pth', '_critic.pth')
            torch.save(self.critic_net.state_dict(), critic_filepath)
    
    def load_model(self, actor_filepath, critic_filepath=None):
        """Load both actor and critic models"""
        self.actor_net.load_state_dict(torch.load(actor_filepath, map_location=self.device))
        self.actor_net.eval()
        
        if critic_filepath is None:
            critic_filepath = actor_filepath.replace('.pth', '_critic.pth')
        
        try:
            self.critic_net.load_state_dict(torch.load(critic_filepath, map_location=self.device))
            self.critic_net.eval()
            print(f"Both models loaded: {actor_filepath}, {critic_filepath}")
        except FileNotFoundError:
            print(f"Actor model loaded: {actor_filepath}")
            print(f"Critic model not found: {critic_filepath}")


# Example usage:
"""
# Create actor and critic networks
state_size = 10  # Example state size
action_size = 3  # Example action size

actor_net = ActorNetwork(state_size, action_size)
critic_net = CriticNetwork(state_size)

# Create trainer
trainer = ActorCriticTrainer(
    actor_net=actor_net,
    critic_net=critic_net,
    lr=0.001,           # Actor learning rate
    critic_lr=0.001,    # Critic learning rate
    gamma=0.99,
    memory_size=10000,
    batch_size=32,
    min_memory_count_to_start_training=1000
)

# Training loop
for episode in range(num_episodes):
    # ... collect experiences ...
    # done=1 if episode ended, done=0 if episode continues
    trainer.store_experience(state, state_plus_1, action, reward, next_state, next_state_plus_1, done)
    
    # Train both networks using TD learning
    losses = trainer.train()
    print(f"Policy Loss: {losses['policy_loss']:.4f}, Value Loss: {losses['value_loss']:.4f}")

# Save models
trainer.save_model("actor_model.pth", "critic_model.pth")
"""
