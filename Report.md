# Report.md

## Deep Deterministic Policy Gradient (DDPG)

### DDPG paper

T. P. Lillicrap et al., 2016. *Continuous control with deep reinforcement learning* (https://arxiv.org/pdf/1509.02971.pdf)

### The basics of DDPG

DDPG trains a deep neural network (**actor**) to learn the policy, i.e., it outputs the actions (whose space is continuous) given the current state. It then trains a second deep neural network (**critic**) to learn the action-value function of the policy. The weights of the actor network is updated by maximizing the action-value that the critic network produces. The weights of the critic network is trained by minimizing the TD error of it's action-values.

Similar to Deep Q-Learning (DQN), DDPG uses a replay buffer to store the experiences. Replays were sampled randomly from the buffer to train both the actor and critic networks. DDPG also uses 2 networks (local and target) for both the actor and the critic networks. The weights of the target networks were updated by the soft-update technique.



### Neural network architecture

The default architectures of the actor (both local and target) consist of:

- **input layer**: a fully connected layer with input size = 33, output size = 400, activation function = ReLU
- **hidden layer**: a fully connected layer with input size = 400, output size = 300, activation function = ReLU
- **output layer**: a fully connected layer with input size = 300, output size = 4, activation function = tanh

The default architectures of the critic (both local and target) consist of:

- **input layer**: a fully connected layer with input size = 33, output size = 400, activation function = ReLU
- **hidden layer**: a fully connected layer with input size = 404, output size = 300, activation function = ReLU
- **output layer**: a fully connected layer with input size = 300, output size = 1

**Notes** In the DDPG paper, the state vector is passed through the input layer alone.
After ReLU activation, the output (size = 400) is concatenated with the action vector (size = 4) and becomes the input (size = 404) to the next hidden layers.

More details can be found in the file `networkModels.py`

### The noise process
The essence of policy-based methods is that, to carry out the exploration, the agent needs to try some new actions that are slightly different from the one prescribed by the current policy. In the case where the action space is continuous, this can simply be achieved by calculating the action by the policy and then adding some noise to it. The magnitude of the noise determines how far away from the current optimal policy the agent wants to explore for putative better policies yielding higher rewards.

In the **DDPG paper**, the authors modeled the noise by the **Ornstein-Uhlenbeck Process** (OU noise)

**why OU noise?**
In a nutshell, compared to the Gaussian noise, which consists of pure diffusion, the Ornstein-Uhlenbeck process has an additional **drift** component. If the direction of the drift is set to be always pointing toward the mean value, it can be pictured as a drag that prevents the noise from diffusing to +/- infinity. (note that in the case of Gaussian noise, as time increases to infinity, the noise will also diffuse to +/- infinity)

The OU noise can be modeled by 2 parameters, `ou_theta` that controls the magnitude of the drift term, and `ou_sigma` that controls the magnitude of the diffusion term. In the **DDPG paper**, the authors used `ou_theta` = 0.15 and `ou_sigma` = 0.20. In this project I used `ou_theta` = 0.15 and `ou_sigma` = 0.10. Moreover, I slowly decreased the scaling factor `ou_scale` of the noise from 1.0, by the decay rate `ou_decay` = 0.995 in each episode.

For more details of the Ornstein-Uhlenbeck Process, please refer to the textbook *Stochastic Methods, a handbook for the natural and social sciences* by Crispin Gardiner (https://www.springer.com/gp/book/9783540707127).


### The DDPG algorithm

Below is a brief description of DDPG:

- initialize the agent with the actor (local and target) and critic (local and target) networks
- t_step = 0
- observe the initial state **s** (size = 33) from the environment
- while the environment is not solved:
  - t_step += 1
  - use *actor_local* to determine the action **a** = actor_local(s)
  - use **a** to interact with the environment
  - collect the reward **r** and enter the next state (**s'**)
  - add the experience tuple **(s, a, r, s')** into the replay buffer
  
  - if there are enough (>= `batch_size`) replays in the buffer:
  
    **step 1: sample replays**
    - randomly sample a batch of size = `batch_size` replay tuples **(s, a, r, s')** from the buffer
    
    **step 2: train `critic_local`**
    - use *actor_target* predict the future actions **a'** = actor_target(s')
    - use *critic_target* to predict the action value of the next state **q_next** = critic_target(s', a')
    - calculate the TD target of the action value **q_target** = **r** + gamma * **q_next**
    - use *critic_local* to calculate the current action value **q_local** = critic_local(s, a)
    - define the loss function (Huber loss) to be the TD error, i.e.,\
    `critic_loss = F.smooth_l1_loss(q_local, q_target)`
    - use gradient decent to update the weights of *critic_local*
    
    **step 3: train `actor_local`**
    - use *actor_local* to determine the action **a_local** = actor_local(s)
    - use *critic_local* to calculate the action values and averaged over the sample batch to obtain the loss of actor:\
    `actor_loss = -critic_local(s, a_local).mean()` (averaged over the batch)
    - use gradient descent to update the weights of *actor_local*
    
    **step 4: update `actor_target` and `critic_target`**
    - soft-update the weights of *actor_target* and *critic_target*\
    `actor_target.weights <-- tau * actor_local.weights + (1-tau) * actor_target.weights`\
    `critic_target.weights <-- tau * critic_local.weights + (1-tau) * critic_target.weights`
  - **s** <-- **s'**


In this project, we have 20 identical agents (robot arms) interacting with the environment. Therefore, at each time step, 20 different experience tuples are added into the buffer. All 20 agents share the same actor and critic networks, whose weights were updated by sampling the replays at each time step (as long as there're enough replays to sample).


## Hyperparameters

| Hyperparameter | Value | Description | Reference |
| ----------- | ----------- | ----------- | ----------- |
| actor_hidden_sizes | [400, 300] | sizes of actor's hidden FC layers | <1> |
| critic_hidden_sizes | [400, 300] | sizes of critic's the hidden FC layers | <1> |
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

<1> Theses are the values used in the original DDPG paper (https://arxiv.org/pdf/1509.02971.pdf)
Please refer to the **Experiment details** in the paper for more information.


## Training result

With the parameters above, the agent solved the task after 0 episodes, i.e., the average (over agents) score from episode #1 to #100 reaches above +30.0 points.

[![p2-scores.png](https://i.postimg.cc/bJ8HzjWw/p2-scores.png)](https://postimg.cc/XZHy8tnR)\
**(figure)** *Average score (over agents) per episodes*

## Ideas for future work

By using the hyperparameters listed above, the agent performs very well on the task. However, I'll spending some more time trying different values of the learning rates of the actor and critic networks to see if better performance can be made. The second thing I'll try out is to change the architecture of the critic network, making it different from the one that DDPG paper used. For example, I'll try passing the state vector and action vector to 2 separate input layers, and concatenate them and pass them together through the second hidden layer.

Besides DDPG, there are many state-of-the-art policy-based algorithm such as **Proximal Policy Optimization (PPO)** and **Deep Distributed Distributional Deterministic Policy Gradient (D4PG)**. I'll implement them after some further reading.

## References in this project
1. T. P. Lillicrap et al., 2016. *Continuous control with deep reinforcement learning*\
https://arxiv.org/pdf/1509.02971.pdf
2. Udacity's GitHub repository **ddpg-pendulum**\
https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
3. Udacity's jupyter notebook template of **Project: Continuous Control**


[![p2-env-demo-trained.png](https://i.postimg.cc/WpSLcNCt/p2-env-demo-trained.png)](https://postimg.cc/5jHkwVxM)\
**(figure)** *Agents have been trained to stay in their target regions!*
