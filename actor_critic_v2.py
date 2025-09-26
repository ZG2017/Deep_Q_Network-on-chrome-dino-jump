"""
Actor-Critic Implementation v2 using GAE and Trajectory-based Learning

This module implements an improved Actor-Critic algorithm that combines:
- Policy network (Actor) for action selection
- Value network (Critic) for state value estimation
- Trajectory-based replay memory system for storing complete episodes
- GAE (Generalized Advantage Estimation) for advantage calculation
- Batch training from latest N trajectories

Key improvements over v1:
1. Replay memory stores complete trajectories instead of single actions
2. Uses GAE to estimate advantages instead of TD error
3. Training samples from latest N trajectories and uses GAE for actor updates
4. Uses MSE loss for critic model updates
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        The Actor network (policy network) with input dim being 3*states_size to handle speed up in later of the game.
        Args:
            state_size: int, the number of states
            action_size: int, the number of actions 
        """
        super(ActorNetwork, self).__init__()
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_size*3, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, action_size),
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
            nn.Linear(state_size*3, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
                
    def forward(self, state_t, state_t_plus_1):
        x = torch.cat([state_t, torch.abs(state_t_plus_1 - state_t), state_t_plus_1], dim=1)
        value = self.value_net(x)
        return value

class ActorNetworkV2(nn.Module):
    def __init__(self, state_size, action_size):
        """
        The Actor network (policy network) with input dim being 3*states_size to handle speed up in later of the game.
        Args:
            state_size: int, the number of states
            action_size: int, the number of actions 
        """
        super(ActorNetworkV2, self).__init__()
        
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

class CriticNetworkV2(nn.Module):
    def __init__(self, state_size):
        """
        Critic network to estimate state values for Actor-Critic architecture.
        Args:
            state_size: int, the number of states
        """
        super(CriticNetworkV2, self).__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(state_size*3, 120),
            nn.ReLU(),
            nn.Linear(120, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
        )
                
    def forward(self, state_t, state_t_plus_1):
        x = torch.cat([state_t, torch.abs(state_t_plus_1 - state_t), state_t_plus_1], dim=1)
        value = self.value_net(x)
        return value



class ActorNetworkV3(nn.Module):
    def __init__(self, state_size, action_size):
        """
        The Actor network (policy network) with input dim being 3*states_size to handle speed up in later of the game.
        Args:
            state_size: int, the number of states
            action_size: int, the number of actions 
        """
        super(ActorNetworkV3, self).__init__()
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_size, 60),
            nn.ReLU(),
            nn.Linear(60, 10),
            nn.ReLU(),
            nn.Linear(10, action_size),
        )
                
    def forward(self, state_t, state_t_plus_1):
        # Policy network
        x = torch.abs(state_t_plus_1 - state_t)
        action_probs = F.softmax(self.policy_net(x), dim=-1)
        return action_probs

class CriticNetworkV3(nn.Module):
    def __init__(self, state_size):
        """
        Critic network to estimate state values for Actor-Critic architecture.
        Args:
            state_size: int, the number of states
        """
        super(CriticNetworkV3, self).__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(state_size, 60),
            nn.ReLU(),
            nn.Linear(60, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
                
    def forward(self, state_t, state_t_plus_1):
        x = torch.abs(state_t_plus_1 - state_t)
        value = self.value_net(x)
        return value

class Trajectory:
    """Class to store a complete trajectory (episode)"""
    def __init__(self):
        self.states_t = []
        self.states_t_plus_1 = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.timestamps = []  # Time from first state
        self.length = 0
        self.start_time = None
    
    def add_step(self, state_t, state_t_plus_1, action, reward, done, current_time=None):
        """Add a single step to the trajectory"""
        if self.start_time is None:
            self.start_time = current_time if current_time is not None else 0
        
        self.states_t.append(state_t.copy())
        self.states_t_plus_1.append(state_t_plus_1.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        
        # Calculate time from first state
        time_from_start = (current_time - self.start_time) if current_time is not None else self.length
        self.timestamps.append(time_from_start)
        self.length += 1
    
    def get_trajectory_data(self):
        """Return trajectory data as numpy arrays"""
        return {
            'states_t': np.array(self.states_t),
            'states_t_plus_1': np.array(self.states_t_plus_1),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'timestamps': np.array(self.timestamps),
            'length': self.length
        }
    
    def is_empty(self):
        """Check if trajectory is empty"""
        return self.length == 0

class TrajectoryReplayMemory:
    """Replay memory that stores complete trajectories instead of single actions.
    Only keeps the longest max_trajectories trajectories and samples the longest ones for training."""
    def __init__(self, max_trajectories=100, max_steps_per_batch=1000):
        self.max_trajectories = max_trajectories
        self.max_steps_per_batch = max_steps_per_batch
        self.trajectories = []  # Use list instead of deque for easier sorting
        self.current_trajectory = Trajectory()
    
    def add_step(self, state_t, state_t_plus_1, action, reward, done, current_time=None):
        """Add a step to the current trajectory"""
        self.current_trajectory.add_step(state_t, state_t_plus_1, action, reward, done, current_time)
        
        # If episode is done, store the complete trajectory
        if done:
            if not self.current_trajectory.is_empty():
                self._add_trajectory(self.current_trajectory)
                self.current_trajectory = Trajectory()
    
    def _add_trajectory(self, trajectory):
        """Add a trajectory to memory, keeping only the longest ones"""
        # Add the new trajectory
        self.trajectories.append(trajectory)
        
        # If we exceed max_trajectories, remove the shortest one
        if len(self.trajectories) > self.max_trajectories:
            # Sort by length (descending) and keep only the longest ones
            self.trajectories.sort(key=lambda traj: traj.length, reverse=True)
            # Remove the shortest trajectory (last in the sorted list)
            self.trajectories.pop()
    
    def get_latest_trajectories(self, n):
        """Get the latest n trajectories (for compatibility)"""
        if len(self.trajectories) == 0:
            return []
        return self.trajectories[-n:]
    
    def sample_trajectories(self, n):
        """Sample trajectories for training, respecting max_steps_per_batch limit"""
        if len(self.trajectories) == 0:
            return []
        
        # Sort trajectories by length (descending) to get the longest ones
        sorted_trajectories = sorted(self.trajectories, key=lambda traj: traj.length, reverse=True)
        
        # Select trajectories while respecting max_steps_per_batch limit
        selected_trajectories = []
        total_steps = 0
        
        for trajectory in sorted_trajectories:
            if len(selected_trajectories) == 0:
                selected_trajectories.append(trajectory)
                total_steps += trajectory.length
            elif total_steps + trajectory.length <= self.max_steps_per_batch:
                # Check if adding this trajectory would exceed the limit
                selected_trajectories.append(trajectory)
                total_steps += trajectory.length
                
                # Stop if we've reached the maximum number of trajectories
                if len(selected_trajectories) >= n:
                    break
            else:
                # If adding this trajectory would exceed the limit, stop
                break
        
        return selected_trajectories
    
    def get_total_steps(self):
        """Get total number of steps across all trajectories"""
        return sum(traj.length for traj in self.trajectories)
    
    def get_memory_stats(self):
        """Get detailed statistics about the memory"""
        if not self.trajectories:
            return {
                'num_trajectories': 0,
                'total_steps': 0,
                'current_trajectory_length': self.current_trajectory.length,
                'longest_trajectory_length': 0,
                'shortest_trajectory_length': 0,
                'average_trajectory_length': 0
            }
        
        lengths = [traj.length for traj in self.trajectories]
        return {
            'num_trajectories': len(self.trajectories),
            'total_steps': sum(lengths),
            'current_trajectory_length': self.current_trajectory.length,
            'longest_trajectory_length': max(lengths),
            'shortest_trajectory_length': min(lengths),
            'average_trajectory_length': sum(lengths) / len(lengths)
        }
    
    def clear(self):
        """Clear all trajectories"""
        self.trajectories.clear()
        self.current_trajectory = Trajectory()

class ActorCriticTrainerV2:
    def __init__(self, actor_net, critic_net, lr=0.001, critic_lr=0.001, gamma=0.99, 
                 lambda_gae=0.95, max_trajectories=100, batch_size=32, 
                 min_trajectories_to_start_training=5, max_steps_per_batch=10000):
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.actor_optimizer = optim.AdamW(actor_net.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(critic_net.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lambda_gae = lambda_gae  # GAE parameter
        self.max_trajectories = max_trajectories
        self.batch_size = batch_size
        self.min_trajectories_to_start_training = min_trajectories_to_start_training
        self.max_steps_per_batch = max_steps_per_batch
        
        # Trajectory-based replay memory
        self.memory = TrajectoryReplayMemory(max_trajectories, max_steps_per_batch)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move networks to device
        self.actor_net.to(self.device)
        self.critic_net.to(self.device)
    
    def add_step(self, state_t, state_t_plus_1, action, reward, done, current_time=None):
        """Add a step to the current trajectory"""
        self.memory.add_step(state_t, state_t_plus_1, action, reward, done, current_time)
    
    def choose_action(self, s_1, s_2):
        """Choose action using the actor network"""
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
        # Sample action from policy
        action_probs_cpu = action_probs.cpu().numpy()[0]
        action = np.argmax(action_probs_cpu)
        return action
    
    def _compute_gae_advantages(self, rewards, values, dones, gamma, lambda_gae):
        """Compute GAE advantages for a trajectory"""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # Compute advantages using GAE
        gae = 0
        gain = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                if dones[t]:
                    delta = rewards[t] - values[t]
                else:
                    delta = rewards[t] + gamma * values[t] - values[t]
            else:
                delta = rewards[t] + gamma * values[t+1] - values[t] # TD error
            gae = delta + gamma * lambda_gae * gae
            gain = rewards[t] + gamma * gain
            advantages[t] = gae
            returns[t] = gain
        
        return advantages, returns
    
    def _sample_data_by_time(self, all_data, sample_ratio=0.8):
        """Sample a subset of data based on time-based probabilities (higher time = higher probability)"""
        if len(all_data['timestamps']) == 0:
            return all_data
        
        timestamps = np.array(all_data['timestamps'])
        max_time = np.max(timestamps)
        
        # Calculate probabilities based on normalized timestamps
        # Higher timestamps get higher probabilities
        if max_time > 0:
            probabilities = timestamps / max_time
        else:
            probabilities = np.ones_like(timestamps)
        
        # Normalize probabilities to sum to 1
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample indices based on probabilities
        n_samples = int(len(timestamps) * sample_ratio)
        if n_samples == 0:
            n_samples = 1
        
        sampled_indices = np.random.choice(
            len(timestamps), 
            size=n_samples, 
            replace=False, 
            p=probabilities
        )
        
        # Create sampled data dictionary
        sampled_data = {}
        for key in all_data.keys():
            if key == 'timestamps':
                sampled_data[key] = timestamps[sampled_indices]
            else:
                sampled_data[key] = np.array(all_data[key])[sampled_indices]
        
        return sampled_data
    
    def train(self):
        """Train both policy and value networks using GAE and trajectory-based learning"""
        # Check if we have enough trajectories to start training
        if len(self.memory.trajectories) < self.min_trajectories_to_start_training:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        # Sample trajectories from the latest ones
        sampled_trajectories = self.memory.sample_trajectories(
            min(self.batch_size, len(self.memory.trajectories))
        )
        
        if not sampled_trajectories:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        # Collect all data from sampled trajectories
        all_states_t = []
        all_states_t_plus_1 = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_timestamps = []
        all_values = []
        all_advantages = []
        all_returns = []
        
        for traj in sampled_trajectories:
            traj_data = traj.get_trajectory_data()

            if traj_data['length'] <= 1:
                continue
            
            # Convert to tensors for value estimation
            states_t_tensor = torch.FloatTensor(traj_data['states_t']).to(self.device)
            states_t_plus_1_tensor = torch.FloatTensor(traj_data['states_t_plus_1']).to(self.device)
            
            # Get value estimates for current and next states
            with torch.no_grad():
                values = self.critic_net(states_t_tensor, states_t_plus_1_tensor).squeeze().cpu().numpy()
            
            # Compute GAE advantages
            advantages, returns = self._compute_gae_advantages(
                traj_data['rewards'], values, traj_data['dones'], 
                self.gamma, self.lambda_gae
            )
            
            # Store data
            all_states_t.extend(traj_data['states_t'])
            all_states_t_plus_1.extend(traj_data['states_t_plus_1'])
            all_actions.extend(traj_data['actions'])
            all_rewards.extend(traj_data['rewards'])
            all_dones.extend(traj_data['dones'])
            all_timestamps.extend(traj_data['timestamps'])
            all_values.extend(values)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
        
        # Prepare data for time-based sampling
        all_data = {
            'states_t': all_states_t,
            'states_t_plus_1': all_states_t_plus_1,
            'actions': all_actions,
            'rewards': all_rewards,
            'dones': all_dones,
            'timestamps': all_timestamps,
            'values': all_values,
            'advantages': all_advantages,
            'returns': all_returns
        }
        
        # Sample data based on time (higher time = higher probability)
        sampled_data = self._sample_data_by_time(all_data, sample_ratio=0.5)
        
        # Convert sampled data to tensors
        states_t_tensor = torch.FloatTensor(np.array(sampled_data['states_t'])).to(self.device)
        states_t_plus_1_tensor = torch.FloatTensor(np.array(sampled_data['states_t_plus_1'])).to(self.device)
        actions_tensor = torch.LongTensor(sampled_data['actions']).to(self.device)
        advantages_tensor = torch.FloatTensor(sampled_data['advantages']).to(self.device)
        returns_tensor = torch.FloatTensor(sampled_data['returns']).to(self.device)
        
        # Normalize advantages for stability
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Forward pass for both networks
        action_probs = self.actor_net(states_t_tensor, states_t_plus_1_tensor)
        state_values = self.critic_net(states_t_tensor, states_t_plus_1_tensor).squeeze()
        
        # Policy loss using GAE advantages
        log_action_probs = torch.log(action_probs + 1e-8)
        selected_log_action_probs = log_action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        policy_loss = -(selected_log_action_probs * advantages_tensor).mean()
        
        # Value loss using MSE between value estimates and returns
        value_loss = F.mse_loss(state_values, returns_tensor)
        
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
    
    def reset_trainer(self, max_trajectories, max_steps_per_batch=None):
        """Reset the trainer with new trajectory memory size"""
        self.max_trajectories = max_trajectories
        if max_steps_per_batch is None:
            max_steps_per_batch = self.max_steps_per_batch
        self.max_steps_per_batch = max_steps_per_batch
        self.memory = TrajectoryReplayMemory(max_trajectories, max_steps_per_batch)
    
    def get_memory_stats(self):
        """Get statistics about the replay memory"""
        return self.memory.get_memory_stats()


# Example usage:
"""
# Create actor and critic networks
state_size = 10  # Example state size
action_size = 3  # Example action size

actor_net = ActorNetwork(state_size, action_size)
critic_net = CriticNetwork(state_size)

# Create trainer v2
trainer = ActorCriticTrainerV2(
    actor_net=actor_net,
    critic_net=critic_net,
    lr=0.001,           # Actor learning rate
    critic_lr=0.001,    # Critic learning rate
    gamma=0.99,
    lambda_gae=0.95,    # GAE parameter
    max_trajectories=100,
    batch_size=32,
    min_trajectories_to_start_training=5,
    max_steps_per_batch=1000  # Maximum steps per training batch
)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Choose action
        action = trainer.choose_action(state, state_plus_1)
        
        # Take action and get next state
        next_state, reward, done, _ = env.step(action)
        
        # Add step to trajectory (automatically handles episode completion)
        trainer.add_step(state, state_plus_1, action, reward, done)
        
        state = next_state
    
    # Train both networks using GAE and trajectory-based learning
    losses = trainer.train()
    print(f"Policy Loss: {losses['policy_loss']:.4f}, Value Loss: {losses['value_loss']:.4f}")
    
    # Print memory statistics
    stats = trainer.get_memory_stats()
    print(f"Memory: {stats['num_trajectories']} trajectories, {stats['total_steps']} total steps")

# Save models
trainer.save_model("actor_model_v2.pth", "critic_model_v2.pth")
"""
