# DeepRL-continuous-control-reachers-udacity-drlnd-p2


- Reinforcement learning environment by Unity ML-Agents
- This repository corresponds to **Project #2 (version 2)** of Udacity's Deep Reinforcement Learning Nanodegree (drlnd)\
  https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
- Continuous control using the actor-critic method

In this project, I used the deep deterministic policy gradient (DDPG) algorithm to train 20 identical agents to move their double-jointed arms to target locations. The goal is to train the agents to remain inside the target locations as long as possible.

The environment is originally from Unity Machine Learning Agents (Unity ML-Agents). For more details and other environments, please visit:\
https://github.com/Unity-Technologies/ml-agents

This project uses the environment provided by Udacity, which is slightly different from the original Unity environment. To run the codes in this repository successfully, Udacity's environment must be used.

[![p2-env-demo.png](https://i.postimg.cc/Rh9ZfxND/p2-env-demo.png)](https://postimg.cc/8JKGQ3dR)\
**(figure)** *The reacher (20 agents version) environment by Unity ML-Agents*

## Project details

- **Number of agents**\
There're 20 identical agents. Each agent controls a double-jointed arm.

- **States**\
The size of the state space is 33 consisting of the arm's position, orientation, velocity and angular velocity.

- **Actions**\
The action has 4 dimensions. The action space is continuous, representing the torques exerted onto its two joints. The value of the action in each dimension can only lie **between -1 and +1**.

- **Rewards**\
For each time step in the episode, a reward of +0.1 is provided if the agent's arm is inside the target region.

- **Goal**\
The environment is considered solved when the average score (**over all 20 agents** and **over 100 consecutive episodes**), reaches above **+30.0**



## Getting started

Please follow the steps below to download all the necessary files and dependencies.

1. Install Anaconda (with Python 3.x)\
    https://www.anaconda.com/products/individual
    
2. Create (if you haven't) a new environment with Python 3.6 by typing the following command in the Anaconda Prompt:\
    `conda create --name drlnd python=3.6`
    
3. Install (need only minimal install) `gym` by following the **Installation** section of the OpenAI Gym GitHub:
    https://github.com/openai/gym#id5
    
4. Clone the repository from Udacity's drlnd GitHub
    ``` console
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```
    (For Windows) If the error "Could not find a version that satisfies the requirement torch==0.4.0 (from unityagents==0.4.0)" occurs, please refer to this thread:\
    https://github.com/udacity/deep-reinforcement-learning/issues/13
  
5. Download the Reacher Environment (Udacity's modified version)\
    Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip \
    Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip \
    Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip \
    Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip
    
    Extract the .zip file and move the folder `Reacher_Windows_x86_64` (or `Reacher_Windows_x86`) into the folder `p2_continuous-control` from Step 4.

6. Download all the files (see the table below) from this repository. Place all files in the folder `p2_continuous-control` from Step 4.

    | File Name | Notes |
    | ----------- | ----------- |
    | SL_reacher20_ddpg.ipynb | main code |
    | ddpgAgent.py | ddpg agent class |
    | networkModels.py | architectures of actor and critic |
    | buffer.py | replay buffer |
    | noiseModels.py | noise process |
    | checkpoint_actor.pth | pre-trained weights for the actor |
    | checkpoint_critic.pth | pre-trained weights for the critic |

7. You're ready to run the code! Please see the next section.

## How to run the code

Please follow the steps below to train the agent or to watch a pre-trained agent perform the task.

#### 1. Run the *Anaconda Prompt* and navigate to the folder `p2_continuous-control`
``` cmd
cd path_to_the_p2_continuous-control_folder
```
#### 2. Activate the drlnd environment
``` cmd
conda activate drlnd
```
#### 3. Run the Jupter Notebook
``` cmd
jupyter notebook
```
#### 4. Open `SL_reacher20_ddpg.ipynb` with Jupyter Notebook
#### 5. Run `Box 1` to import packages
Paste the path to `Reacher.exe` after **"file_name = "**\
for example, `file_name = "./Reacher_20Agent_Windows_x86_64/Reacher.exe"`
#### 6. Run `Box 2` to set the hyperparameters
For information of the hyperparameters, please refer to `Report.md`
#### 7. Run `Box 3` to start training the agent
A figure of the noise simulation will be displayed first, which can be used for tuning the parameters of the noise process. Please note that this simulation is independent of the noise process used in the ddpg traning, i.e., the exact noise values generated during training will be different from the values shown in the figure.
After training, the weights and biases of the actor and critic networks will be saved with the file names `checkpoint_actor.pth` and `checkpoint_critic.pth`
#### 8. (Optional) Run `Box 4` to load the saved weights into the agent and watch the performance
#### 9. Before closing, simply use the command `env.close()` to close the environment
