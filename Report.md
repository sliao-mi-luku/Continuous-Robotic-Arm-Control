# Report.md

# THIS IS STILL A MANUSCRIPT

## Deep Deterministic Policy Gradient (DDPG)

### DDPG paper (*T. P. Lillicrap et al., 2016*)

Continuous control with deep reinforcement learning

https://arxiv.org/pdf/1509.02971.pdf

### The basics of DDPG


### Neural network architecture

The default architectures of the actor (both local and target) consist of:

- **input layer**: a fully connected layer with input size = 33, output size = 400, activation function = ReLU
- **hidden layer**: a fully connected layer with input size = 400, output size = 300, activation function = ReLU
- **output layer**: a fully connected layer with input size = 300, output size = 4

The default architectures of the critic (both local and target) consist of:

- **input layer**: a fully connected layer with input size = 33, output size = 400, activation function = ReLU
- **hidden layer**: a fully connected layer with input size = 404, output size = 300, activation function = ReLU
- **output layer**: a fully connected layer with input size = 300, output size = 1

**Notes** In the DDPG paper, the state vector is passed through the input layer alone.
After ReLU activation, the output (size = 400) is concatenated with the action vector (size = 4) and becomes the input (size = 404) to the next hidden layers.

More details can be found in the file `networkModels.py`

### The noise process
The essence of policy-based methods is that, in the sense of exploration, the agent needs to try some new actions that are slightly different from the one prescribed by the current policy. This can simply be achieved by obtaining the action by the policy and then adding some noise to it. The magnitude of the noise determines how far away from the current optimal policy the agent wants to explore for putative better policies that can yield better reward.

In the **DDPG paper**, the authors modeled the noise by the **Ornstein-Uhlenbeck Process** (OU noise)

**why OU noise?**
In a nutshell, compared to the Gaussian noise, which is pure diffusion, the Ornstein-Uhlenbeck process has an additional **drift** term. If the direction of the drift is always pointing toward the initial value, it can be pictured as a drag toward to the initial value that prevents the noise to diffuse toward +/- infinity.

This is also manifesting the drawback of using Gaussian noise, as time increases to infinity, the Gaussian process goes to +/- infinity.

The OU noise can be modeled by 2 parameters, `ou_theta` that controls the size of the drift, and `ou_sigma` that controls the size of diffusion. In the **DDPG paper**, the authors used `ou_theta` = 0.15 and `ou_sigma` = 0.20. In this project I used `ou_theta` = 0.15 and `ou_sigma` = 0.10. Furthermore, I slowly decreased the scaling facor `ou_scale` of the OU noise from 1 (initial scaling) by the decay rate `ou_decay` = 0.995 for each episode.

For more details on the Ornstein-Uhlenbeck Process, please refer to the textbook *Stochastic Methods, a handbook for the natural and social sciences* by Crispin Gardiner (https://www.springer.com/gp/book/9783540707127).


### The DDPG algorithm

Below is a brief description of DDPG:

- initialize the agent with the actor (local and target) and critic (local and target) networks
  > we call them `actor_local`, `actor_target`, `critic_local` and `critic_target`
  > `actor_local` and `actor_target` will have the same initial weights, and so do `critic_local` and `critic_target`
- t_step = 0
- observe the initial state **s** (size = 33) from the environment
- while the environment is not solved:
  - t_step += 1
  - use `actor_local` to determine the action **a**  = `actor_local(s)`
  - use **a** to interact with the environment
  - collect the reward **r** and enter the next state (**s'**)
  - add the experience tuple **(s, a, r, s')** into the replay buffer
  
  - if there are enough (>= `batch_size`) replays in the buffer:
  
    **step 1: sample replays**
    - randomly sample a batch of size = `batch_size` replay tuples **(s, a, r, s')** from the buffer\
    
    **step 2: train `critic_local`**
    - use `actor_target` predict the future actions **a'** = actor_target(s')
    - use `critic_target` to predict the action value of the next state **q_next** = `critic_target(s', a')`
    - calculate the TD target of the action value **q_target** = **r** + gamma * **q_next**
    - use `critic_local` to calculate the current action value **q_local** = `critic_local(s, a)`
    - define the loss function (Huber loss) to be the TD error, i.e.,\
    `critic_loss = F.smooth_l1_loss(**q_local**, *q_target*)`
    - use gradient decent to update the weights of *critic_local*
    
    **step 3: train `actor_local`**
    - use `actor_local` to determine the action `a_local = actor_local(s)`
    - use `critic_local` to calculate the action values and averaged over the sample batch to obtain the loss of actor:\
    `actor_loss = -critic_local(s, a_local) (averaged over the batch)`
    - use gradient descent to update the weights of *actor_local*
    
    **step 4: update `actor_target` and `critic_target`**
    - soft-update the weights of `actor_target` and `critic_target`\
    `actor_target.weights <-- tau * actor_local.weights + (1-tau) * actor_target.weights`
    `critic_target.weights <-- tau * critic_local.weights + (1-tau) * critic_target.weights`
  - s <-- s'


## Hyperparameters

| Hyperparameter | Value | Description | Reference |
| ----------- | ----------- | ----------- | ----------- |
| actor_hidden_sizes | [400, 300] | sizes of the hidden FC layers of actor | <1> |
| critic_hidden_sizes | [400, 300] | sizes of the hidden FC layers of actor | <1> |
| gamma | 0.99 | discount rate | <1> |
| actor_lr | 1e-3 | actor's learning rate |  |
| critic_lr | 1e-4 | critic's learning rate |  |
| critic_L2_decay | 0 | critic's L2 weight decay |  |
| tau | 1e-3 | soft update | <1> |
| ou_scale | 1.0 | initial scaling of noise |  |
| ou_decay | 0.995 | scaling decay of noise |  |
| ou_mu | 0.0 | initial mean of noise | <1> |
| ou_theta | 0.15 | drift component of noise | <1> |
| ou_sigma | 0.10 | diffusion component of noise |  |
| buffer_size | 1e6 | size of the replay buffer | <1> |
| batch_size | 512 | size of the minibatch |  |

<1> Theses are the values used in the original DDPG paper\
https://arxiv.org/pdf/1509.02971.pdf \
Please refer to the **Experiment details** in the paper for more information.


## Training result

With the parameters above, the agent solved the task after 0 episodes, i.e., the average (over agents) score from episode #1 to #100 reaches above +30.0 points.

[![p2-scores.png](https://i.postimg.cc/bJ8HzjWw/p2-scores.png)](https://postimg.cc/XZHy8tnR)

## Ideas for future work

**1. Promimal Policy Optimization (PPO)**

**2. Asynchronous Advantage Actor-Critic (A3C)**

[![p2-env-demo-trained.png](https://i.postimg.cc/WpSLcNCt/p2-env-demo-trained.png)](https://postimg.cc/5jHkwVxM)
