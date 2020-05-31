import matplotlib.pyplot as plt
import numpy as np

class Info:
  def __init__(self, env):
    self.env = env
    self.brain_name = env.brain_names[0]
    self.brain = env.brains[self.brain_name]
    print("created Info")

  def print_info(self):
    # reset the environment
    env_info = self.env.reset(train_mode=True)[self.brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = self.brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

  def getInfo(self):
    env_info = self.env.reset(train_mode=True)[self.brain_name]
    action_size = self.brain.vector_action_space_size
    # state_size = self.brain.vector_observation_space_size * self.brain.num_stacked_vector_observations
    state_size = env_info.vector_observations.shape[1]
    num_agents = len(env_info.agents)
    return action_size, state_size, num_agents
  
  def plotResults(self, scores, avgs, scores_path="scores.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, label='DDPG')
    plt.plot(np.arange(len(scores)), avgs, c='r', label='moving avg')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(loc='upper left');
    fig.savefig(scores_path)
    plt.show()