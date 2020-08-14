"""
Reference:

This file was modified from ddpg_agent.py from Udacity's GitHub Repository ddpg-pendulum

https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py
"""


## DDPG Agent
import random
import torch
import numpy as np
import torch.optim as optim
from networkModels import ActorNetwork, CriticNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG_Agent:
    
    def __init__(self, actor_input_size = 33, actor_output_size = 4, actor_hidden_sizes = [400, 300],
                critic_state_size = 33, critic_action_size = 4, critic_hidden_sizes = [400, 300],
                actor_lr = 1e-4, critic_lr = 1e-3, critic_L2_decay = 1e-2, 
                agent_idx = 0, seed = 0):

        self.seed = random.seed(seed)
        
        # Actors (local & target)
        self.actor_local = ActorNetwork(actor_input_size, actor_output_size, actor_hidden_sizes, seed).to(device)
        self.actor_target = ActorNetwork(actor_input_size, actor_output_size, actor_hidden_sizes, seed).to(device)
        
        for local_param, target_param in zip(self.actor_local.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(local_param.data)
        
        # Critics (local & target)
        self.critic_local = CriticNetwork(critic_state_size, critic_action_size, critic_hidden_sizes, seed).to(device)
        self.critic_target = CriticNetwork(critic_state_size, critic_action_size, critic_hidden_sizes, seed).to(device)
        
        for local_param, target_param in zip(self.critic_local.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(local_param.data)
        
        # optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = critic_lr, weight_decay = critic_L2_decay)
        
    def local_act(self, state_t, noise_np):
        """
            state_t: state tensor of shape (m, 33)
            noise_np - ndarray of shape (m, 4)
        """
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state_t).cpu().data.numpy()
        self.actor_local.train()
        action += noise_np
        return np.clip(action, -1, 1)
    
    def target_act(self, state_t, noise_np):
        """
            state_t: state tensor of shape (m, 33)
            noise_np - ndarray of shape (m, 4)
        """
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(state_t).cpu().data.numpy()
        self.actor_target.train()
        action += noise_np
        return np.clip(action, -1, 1)
    
    def noisefree_local_act(self, state_t):
        """
            state_t: state tensor of shape (m, 33)
            ** note that the ouput will be a tensor, np ndarray
        """
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state_t)
        self.actor_local.train()
        return action
    
    def reset(self):
        self.noise.reset()