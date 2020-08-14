# Report.md

# THIS IS STILL A MANUSCRIPT

## Deep Deterministic Policy Gradient (DDPG)

**The original DDPG paper**

https://arxiv.org/pdf/1509.02971.pdf

### The basics

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
In the DDPG paper, the noise process is simulated by the Ornstein-Uhlenbeck Process

**What is OU noise?**


### The DDPG algorithm

Below is a brief description of DDPG:

- initialize the agent with the actor (local and target) and critic (local and target) networks
- t_step = 0
- observe the initial state **s** (size = 33) from the environment
- while the environment is not solved:
  - t_step += 1
  - use *actor_local* to calculate the action (for all 4 actions)
  - the agent uses the epsilon-greedy policy to take an action **a**
  - the agent collects the reward **r** and enters the next state (**s'**)
  - add the experience tuple **(s, a, r, s')** into the replay buffer
  - if there are enough (>= `batch_size`) replays in the buffer:
    - randomly sample a batch of size = `batch_size` replay tuples **(s, a, r, s')** from the buffer
    - use *Q_target* to calculate predicted action value `max_{a'}Q_target(s', a')`
    - use *Q_local* to calculate action value Q_local(s, a)
    - use gradient decent (loss = MSE) to update the weights of *Q_local* by minimizing the error\
    `r + gamma * max_{a'}Q_target(s', a') - Q_local(s, a)`
    - soft-update the weights of *Q_target*\
    `new_Q_target_weights <-- tau * old_Q_local_weights + (1-tau) * old_Q_target_weights`
  - s <-- s'


## Hyperparameters

| Hyperparameter | Value | Description | Reference |
| ----------- | ----------- | ----------- | ----------- |
| hidden_sizes | [400, 300] | input and output sizes of the hidden FC layers | |
| gamma | 0.99 | discount rate | |
| lr | 1e-4 | learning rate | |
| tau | 1e-3 | soft update rate <1> | |
| UPDATE_EVERY | 10 | frequency of updating Q network | |
| eps_start | 1.0 | initial epsilon | |
| eps_end | 0.01 | terminal epsilon | |
| eps_decay | 0.995 | epsilon decay rate <2> | |
| buffer_size | 1e5 | size of the replay buffer | |
| batch_size | 64 | number of samples to train the Q-network each time | |

**<1> (Soft update)** This algorithm slowly updates the target network every `UPDATE_EVERY` steps in an episodes by the soft updating rule:\
`new_target_weights <-- tau * old_local_weights + (1-tau) * old_target_weights`


## Training result

With the parameters above, the agent solved the task after 0 episodes, i.e., the average (over agents) score from episode #1 to #100 reaches above +30.0 points.

[![p2-scores.png](https://i.postimg.cc/bJ8HzjWw/p2-scores.png)](https://postimg.cc/XZHy8tnR)

## Ideas for future work


[![p2-env-demo-trained.png](https://i.postimg.cc/WpSLcNCt/p2-env-demo-trained.png)](https://postimg.cc/5jHkwVxM)
