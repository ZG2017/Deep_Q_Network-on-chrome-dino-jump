"""
REINFORCE Implementation

This module implements the REINFORCE algorithm, which is a policy gradient method
that learns a policy directly without using a value function. It uses Monte Carlo
returns to estimate the policy gradient.

REINFORCE is an on-policy algorithm that:
- Learns a stochastic policy π(a|s)
- Uses Monte Carlo returns for policy gradient estimation
- Trains after each complete episode
- Has high variance but is simple and direct
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class REINFORCE(nn.Module):
    def __init__(self, state_size, action_size):
        """
        The REINFORCE model with input dim being 3*states_size to handle speed up in later of the game.
        Args:
            state_size: int, the number of states
            action_size: int, the number of actions 
        """
        super(REINFORCE, self).__init__()
        
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

class REINFORCETrainer:
    def __init__(self, policy_net, lr=0.001, gamma=0.99):
        self.policy_net = policy_net
        self.optimizer = optim.AdamW(policy_net.parameters(), lr=lr)
        self.gamma = gamma
        
        # Experience buffer for episode-based training
        self.states = []
        self.states_plus_1 = []
        self.actions = []
        self.rewards = []
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def choose_action(self, s_1, s_2):
        s_1 = torch.FloatTensor(s_1).view(1, -1).to(self.device)
        s_2 = torch.FloatTensor(s_2).view(1, -1).to(self.device)
        with torch.no_grad():
            action_probs = self.policy_net(s_1, s_2)
        # Sample action from policy
        action_probs_cpu = action_probs.cpu().numpy()[0]
        action = np.random.choice(len(action_probs_cpu), p=action_probs_cpu)
        return action
    
    def store_experience(self, state, state_plus_1, action, reward):
        """Store experience for training"""
        self.states.append(state)
        self.states_plus_1.append(state_plus_1)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def compute_returns(self, rewards, gamma):
        """Compute discounted returns using Monte Carlo method"""
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + gamma * R
            returns.append(R)
        return torch.FloatTensor(returns[::-1]).to(self.device)
    
    def train(self):
        """Train the policy network using REINFORCE algorithm"""
        if len(self.states) == 0:
            return {'policy_loss': 0.0}
            
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        states_plus_1 = torch.FloatTensor(np.array(self.states_plus_1)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = self.rewards
        
        # Compute returns using Monte Carlo
        returns = self.compute_returns(rewards, self.gamma)
        
        # Forward pass
        action_probs = self.policy_net(states, states_plus_1)
        
        # Policy loss using returns (REINFORCE)
        log_action_probs = torch.log(action_probs + 1e-8)
        selected_log_action_probs = log_action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(selected_log_action_probs * returns).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Clear experience buffer
        self.states = []
        self.states_plus_1 = []
        self.actions = []
        self.rewards = []
        
        return {
            'policy_loss': policy_loss.item()
        }
    
    def save_model(self, filepath):
        """Save the model"""
        torch.save(self.policy_net.state_dict(), filepath)
    
    def load_model(self, filepath):
        """Load the model"""
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.policy_net.eval()
        print(f"Model loaded: {filepath}")


# Example usage:
"""
# Create policy network
state_size = 10  # Example state size
action_size = 3  # Example action size

policy_net = REINFORCE(state_size, action_size)

# Create trainer
trainer = REINFORCETrainer(
    policy_net=policy_net,
    lr=0.001,
    gamma=0.99
)

# Training loop
for episode in range(num_episodes):
    # Collect episode experiences
    for step in range(max_steps):
        action = trainer.choose_action(state, state_plus_1)
        # ... execute action and get reward ...
        trainer.store_experience(state, state_plus_1, action, reward)
    
    # Train after each episode
    loss_info = trainer.train()
    print(f"Policy Loss: {loss_info['policy_loss']:.4f}")

# Save model
trainer.save_model("reinforce_model.pth")
"""