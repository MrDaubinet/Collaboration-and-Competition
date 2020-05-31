import numpy as np

class Baseline:
	def __init__(self, env, action_size, state_size):
		self.env = env
		self.action_size = action_size
		self.state_size = state_size

	def run(self, max_t=1000):
		brain_name = self.env.brain_names[0]
		brain = self.env.brains[brain_name]
		env_info = self.env.reset(train_mode=False)[brain_name]     # reset the environment    
		num_agents = len(env_info.agents)
		states = env_info.vector_observations                  # get the current state (for each agent)
		scores = np.zeros(num_agents)                          # initialize the score (for each agent)
		for t in range(max_t):
			actions = np.random.randn(num_agents, self.action_size) # select an action (for each agent)
			actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
			env_info = self.env.step(actions)[brain_name]           # send all actions to tne environment
			next_states = env_info.vector_observations         # get next state (for each agent)
			rewards = env_info.rewards                         # get reward (for each agent)
			dones = env_info.local_done                        # see if episode finished
			scores += env_info.rewards                         # update the score (for each agent)
			states = next_states                               # roll over states to next time step

		print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
		return scores
