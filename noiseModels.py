"""
Reference:

This file was modified from ddpg_agent.py from Udacity's GitHub Repository ddpg-pendulum

https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py
"""

## Ornstein–Uhlenbeck Process

import numpy as np
import matplotlib.pyplot as plt


class OUNoise:
    def __init__(self, output_size = 4, mu = 0.0, theta = 0.15, sigma = 0.20, seed = 0):
        
        self.seed = np.random.seed(seed)
        
        self.output_size = output_size
        self.mu = mu*np.ones(output_size)
        self.theta = theta
        self.sigma = sigma

        self.reset()

    def reset(self):
        self.x = np.copy(self.mu)

    def get_noise(self):
        self.x += self.theta*(self.mu - self.x) + self.sigma*np.random.rand(self.output_size)
        return self.x
    
    
def plot_OU(t_max = 1000, A0 = 1, decay_factor = 1, output_size = 4, mu = 0.0, theta = 0.15, sigma = 0.20, seed = 0):
    OU = OUNoise(output_size, mu, theta, sigma, seed)
    t = list(range(t_max))
    x = []
    A = A0
    for _ in t:
        x.append(np.mean(A*OU.get_noise()))
        A *= decay_factor
    plt.figure()
    plt.plot(t, x)
    plt.title('Simulated noise process')
    plt.xlabel('Episodes')
    plt.ylabel('|Noise|')
    plt.show()