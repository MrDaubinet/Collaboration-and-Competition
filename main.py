from unityagents import UnityEnvironment
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from info import Info

from baseline import Baseline
from agent import Agent
from ddpg import DDPG
from maddpg import MADDPG

#  Replace with your location of the reacher
env = UnityEnvironment(file_name='C:/Udacity/Deep Reinforcement Learning/deep-reinforcement-learning/p3_collab-compet/Tennis_Windows_x86_64/Tennis.exe')

# Info
info = Info(env) # create the info object
info.print_info() # print out information

# TO NOTE:
# Each observation is a stack of 3 states. 
# Each game state has 8 variables
# making each observation have the size 24

# set action and state
action_size, state_size, num_agents = info.getInfo()

# baseline = Baseline(env, action_size, state_size)
# baseline.run()
seed = 6

random.seed(seed)
torch.manual_seed(seed)

# Create the maddpg object
maddpg = MADDPG(env, state_size, action_size, num_agents, seed)

# train agent
# scores, average_scores_list = maddpg.train(n_episodes=5000)

# info.plotResults(scores, average_scores_list) # plot the scores

# test best agent
maddpg.test(env, state_size)


