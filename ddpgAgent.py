"""
Reference:

This file was modified from ddpg_agent.py from Udacity's GitHub Repository ddpg-pendulum

https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py
"""

import torch
import numpy as np
import torch.optim as optim
from networkModels import ActorNetwork, CriticNetwork


## DDPG Agent

class DDPG_Agent:
    
    """
    DDPG agent includes 2 actor (actor_local, actor_target) and 2 critic (critic_local, critic_target) networks
    """
    
    def __init__(self, config):
        
        """
        config: the dictionary containing the keys:
            actor_input_size: input size of the actor (33, dimension of the state)
            actor_output_size: output size of the actor (4, dimension of the action)
            actor_hidden_sizes: input and output sizes of the hidden FC layer of the actor
            
            critic_state_size: dimension of the state (33)
            critic_action_size: dimension of the action (4)
            critic_hidden_sizes: input and output sizes of the hidden FC layer of the critic
            
            actor_lr: learning rate of the actor
            critic_lr: learning rate of the critic
            critic_L2_decay: L2 weight decay of the critic
        """
        actor_input_size = config['actor_input_size']
        actor_output_size = config['actor_output_size']
        actor_hidden_sizes = config['actor_hidden_sizes']
        
        critic_state_size = config['critic_state_size']
        critic_action_size = config['critic_action_size']
        critic_hidden_sizes = config['critic_hidden_sizes']
        
        actor_lr = config['actor_lr']
        critic_lr = config['critic_lr']
        critic_L2_decay = config['critic_L2_decay']
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Actors (local & target)
        self.actor_local = ActorNetwork(actor_input_size, actor_output_size, actor_hidden_sizes).to(self.device)
        self.actor_target = ActorNetwork(actor_input_size, actor_output_size, actor_hidden_sizes).to(self.device)
        
        # set actor_local and actor_target with same weights & biases
        for local_param, target_param in zip(self.actor_local.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(local_param.data)
        
        # Critics (local & target)
        self.critic_local = CriticNetwork(critic_state_size, critic_action_size, critic_hidden_sizes).to(self.device)
        self.critic_target = CriticNetwork(critic_state_size, critic_action_size, critic_hidden_sizes).to(self.device)
        
        # set critic_local and critic_target with same weights & biases
        for local_param, target_param in zip(self.critic_local.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(local_param.data)
        
        # optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = critic_lr, weight_decay = critic_L2_decay)
        
        
    def act(self, state_t, actor_name = 'target', noise_bool = False, noise_np = None):
        """
        Use the actor network to determine actions
            
        inputs:
            state_t: state tensor of shape (m, 33)
            actor_name: the actor network to use ("local" or "target")
            noise_bool: whether or not to add the noise
            noise_np - the noise to be added (if noise_bool == True), an ndarray of shape (m, 4)

        output:
            the action tensor
        """
        
        if actor_name == 'local':
            actor_network = self.actor_local
        elif actor_name == 'target':
            actor_network = self.actor_target
        
        actor_network.eval()
        
        with torch.no_grad():
            action = actor_network(state_t) # action is a tensor
        
            if noise_bool: # to add noise
                action = action.cpu().data.numpy() # convert action to ndarray
                action = np.clip(action + noise_np, -1, 1) # add noise and clip between [-1, +1]
                action = torch.from_numpy(action).float().to(self.device) # convert action to tensor
            
        actor_network.train()
        
        return action       
