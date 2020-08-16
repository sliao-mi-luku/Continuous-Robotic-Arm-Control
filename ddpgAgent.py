"""
Reference:

This file was modified from ddpg_agent.py from Udacity's GitHub Repository ddpg-pendulum

https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py
"""

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
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
            
            gamma: the discounting rate
            
            tau: soft-update factor
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
        
        self.gamma = config['gamma']
        
        self.tau = config['tau']
        
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
        
        self.current_critic_loss = 0
        self.current_actor_loss = 0
        
    def act(self, state_t, actor_name = 'target', noise_bool = False, noise_np = None):
        """
        Use the actor network to determine actions
            
        inputs:
            state_t: state tensor of shape (m, 33)
            actor_name: the actor network to use ("local" or "target")
            noise_bool: whether or not to add the noise
            noise_np - the noise to be added (if noise_bool == True), an ndarray of shape (m, 4)

        output:
            the action (tensor)
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
    
    
    def learn(self, replays):
        """        
        Used the sampled replays to train the actor and the critic
        
        replays: replay tuples in the format of (states, actions, rewards, next_states, dones)
        """
                        
        ## assign s, a, r, s', d from replays
        states, actions, rewards, next_states, dones = replays

        # size of the batch
        m = actions.shape[0]
        
        # convert from ndarrays to tensors
        states = torch.from_numpy(states).float().to(self.device) # [m, 33]
        actions = torch.from_numpy(actions).float().to(self.device) # [m, 4]
        next_states = torch.from_numpy(next_states).float().to(self.device) # [m, 33]
        rewards = torch.from_numpy(rewards).float().to(self.device) # [m, 1]
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device) # [m, 1]
        
        
        """ Train critic_local """
        # next_actions: use actor_target to obtain next_actions (tensor) from next_states (tensor)
        next_actions = self.act(state_t = next_states, actor_name = 'target', noise_bool = False)
        
        # q_next: use critic_target to obatin the action-value of (next_states, next_actions)
        self.critic_target.eval()
        with torch.no_grad():
            q_next = self.critic_target(next_states, next_actions).detach().to(self.device) # [m, 1]
        self.critic_target.train()
        
        # q_target: the TD target of the critic, i.e. q_target = r + gamma*q_next
        q_target = rewards + self.gamma*q_next*(1-dones) # [m, 1]
        
        # q_local: the current action-value of (states, actions)
        q_local = self.critic_local(states, actions) # [m, 1]

        # critic_loss
        self.critic_optimizer.zero_grad()
        critic_loss = F.smooth_l1_loss(q_local, q_target.detach())
        self.current_critic_loss = critic_loss.data.detach().cpu().numpy()
        critic_loss.backward()
        self.critic_optimizer.step()

        
        """ Train actor_local """
        # local_actions: use actor_local to obain local_actions
        local_actions = self.actor_local(states)  # [m, 4]
        
        # actor_loss
        actor_loss = - self.critic_local(states, local_actions).mean()
        self.current_actor_loss = actor_loss.data.detach().cpu().numpy()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        
        ## soft-update actor_target and critic_target
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)
    
    
    def soft_update(self, local_nn, target_nn):
        """
        Soft-update the weight of the actor (or critic) target network
        """
        for local_param, target_param in zip(local_nn.parameters(), target_nn.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)    
