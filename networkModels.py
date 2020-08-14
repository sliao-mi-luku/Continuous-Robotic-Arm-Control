"""
Reference:

This file was modified from model.py from Udacity's GitHub Repository ddpg-pendulum

https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## Actor Neural Network

class ActorNetwork(nn.Module):
    def __init__(self, input_size = 33, output_size = 4, hidden_sizes = [400, 300], seed = 0):

        super(ActorNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        
        self.hidden_layers = nn.ModuleList([])
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))

        for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self.hidden_layers.append(nn.Linear(h1, h2))
            
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.hidden_layers:
            # fan_in
            f = layer.weight.data.size()[0]
            layer.weight.data.uniform_(-1.0/np.sqrt(f), 1.0/np.sqrt(f))
            layer.bias.data.fill_(0.1)
            
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.output_layer.bias.data.fill_(0.1)
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return F.tanh(self.output_layer(x))



## Critic Neural Network

class CriticNetwork(nn.Module):
    
    def __init__(self, state_size = 33, action_size = 4, hidden_sizes = [400, 300], seed = 0):
        
        super(CriticNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)

        self.first_layer = nn.Linear(state_size, hidden_sizes[0])
        self.second_layer = nn.Linear(hidden_sizes[0] + action_size, hidden_sizes[1])
        self.output_layer = nn.Linear(hidden_sizes[1], 1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        f1 = self.first_layer.weight.data.size()[0]
        f2 = self.second_layer.weight.data.size()[0]

        self.first_layer.weight.data.uniform_(-1.0/np.sqrt(f1), 1.0/np.sqrt(f1))
        self.second_layer.weight.data.uniform_(-1.0/np.sqrt(f2), 1.0/np.sqrt(f2))

        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        
        self.first_layer.bias.data.fill_(0.1)
        self.second_layer.bias.data.fill_(0.1)
        self.output_layer.bias.data.fill_(0.1)
        
    def forward(self, state, action):
        xs = F.relu(self.first_layer(state))
        x = torch.cat((xs, action), dim = 1)
        x = F.relu(self.second_layer(x))
        return self.output_layer(x)