# DeepRL-continuous-control-reachers-udacity-drlnd-p2


- Reinforcement learning environment by Unity ML-Agents
- This repository corresponds to **Project #2 (version 2)** of Udacity's Deep Reinforcement Learning Nanodegree (drlnd)\
  https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
- Continuous control using the actor-critic method

In this project, we used the deep deterministic policy gradient (DDPG) algorithm to train 20 identical agents to move their double-jointed arms to target locations. The goal is to train the agents to remain inside the target locations as long as possible.

The environment is originally from Unity Machine Learning Agents (Unity ML-Agents). For more details and other learning environments, please visit:\
https://github.com/Unity-Technologies/ml-agents

For this project we use a slightly different environment provided by Udacity drlnd.

[![p2-env-demo.png](https://i.postimg.cc/Rh9ZfxND/p2-env-demo.png)](https://postimg.cc/8JKGQ3dR)\
**(figure)** *The reacher (20 agents version) environment by Unity ML-Agents*

## Project details

- **Number of agents**\
There're 20 identical agents. Each agent controls a double-jointed arm.

- **States**\
The size of the state space is 33 consisting of the arm's position, orientation, velocity and angular velocity.

- **Actions**\
The action space has 4 dimensions (continuous), representing the torque vectors exerted onto its 2 joints. The value in each dimension can only lie **between -1 and +1**.

- **Rewards**\
For each time step in an episode, a reward of +0.1 is provided if the agent's arm is inside the target region.

- **Goal**\
The environment is considered solved when the score, **averaged over all 20 agents** and **averaged over 100 consecutive episodes**, reaches above **+30.0**



## Getting started
1. Install Anaconda (with Python 3.x)\
    https://www.anaconda.com/products/individual
    
2. Create (if you haven't) a new environment with Python 3.6 by typing the following command in 
    `conda create --name drlnd python=3.6`
    
3. Install (minimal install) `gym` by following the **Installation** of the openai GitHub:\
    https://github.com/openai/gym#id5
    
4. Clone the Deep-Reinforcement-Learning-Nanodegree GitHub repository
    ``` console
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```
> (For Windows 10) If the error "Could not find a version that satisfies the requirement torch==0.4.0 (from unityagents==0.4.0)" occurs, please refer to this thread:\
    https://github.com/udacity/deep-reinforcement-learning/issues/13
  
5. Download Unity's Reacher Environment or (Udacity's modified verson)\
    Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip \
    Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip
    Extract the .zip file and move the folder `Reacher_Windows_x86_64` (or `Reacher_Windows_x86`) into the folder `p2_continuous-control` from Step 4 above.

6. Download all the files (see the table below) from this repository. Place all files in the folder `p2_continuous-control` from Step 4.

    | File Name | Notes |
    | ----------- | ----------- |
    | SL_reacher20_ddpg.ipynb | the main code |
    | networkModels.py | the architectures of actor and critic |
    | buffer.py | the replay buffer |
    | ddpgAgent.py | the agent class |
    | checkpoint_actor.pth | the saved weights for actor |
    | checkpoint_critic.pth | the saved weights for critic |

7. You're ready to run the code! Please see the next section.

## How to run the code

#### 1. Run the Anaconda Prompt and navigate to the folder `p2_continuous-control`
#### 2. Activate the drlnd environment
``` python
conda activate drlnd
```
#### 3. Run the Jupter Notebook by the command:
``` cmd
jupyter notebook
```
#### 4. Open `SL_reacher20_ddpg.ipynb` with Jupyter Notebook
#### 5. Run `Box 1` to import packages
Paste the path to Reacher.exe after the "file_name = "\
for example, file_name = "./Reacher_20Agent_Windows_x86_64/Reacher.exe"
#### 6. Run `Box 2` to set hyperparameters
For details of the hyperparameters, please refer to `Report.md`
#### 7. Run `Box 3` to start training
A figure of noise simulation will be displayed first, which can be used for tuning the hypermeters of the noise process.\
After training, the weights of the actor and critic will be saved with the file names `checkpoint_actor.pth` and `checkpoint_critic.pth`
#### 8. (Optional) Run `Box 4` to load the saved weights into the agent and watch the performance
#### 9. Before closing, simply use the command `env.close()` to close the environment
