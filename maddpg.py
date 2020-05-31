from collections import namedtuple, deque
import random
import numpy as np
import torch
import time

from agent import Agent


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Cuda available ?")
print(torch.cuda.is_available())

class MADDPG:
    """Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"""
    def __init__(self, env, state_size, action_size, num_agents, random_seed):
        """Initialize an MADDPG object.
        Params
        ======
            random_seed (int): random seed
        """
        self.env = env
        self.action_size = action_size
        self.num_agents = num_agents
        self.brain_name = env.brain_names[0]
        self.timestep = 0

        self.agents = [Agent(state_size, action_size, num_agents, random_seed) for x in range(num_agents)]
        self.shared_memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, num_agents, random_seed)

    def step(self, states, actions, rewards, next_states, dones, t):
        self.shared_memory.add(states, actions, rewards, next_states, dones)

        for agent in self.agents:
            agent.step(self.shared_memory, t)

    def act(self, states, add_noise=True):
        actions = np.zeros([self.num_agents, self.action_size])
        for index, agent in enumerate(self.agents):
            actions[index, :] = agent.act(states[index], add_noise)
        return actions

    def save_weights(self):
        for index, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'agent{}_checkpoint_actor.pth'.format(index+1))
            torch.save(agent.critic_local.state_dict(), 'agent{}_checkpoint_critic.pth'.format(index+1))
    
    def load_weights(self):
        for index, agent in enumerate(self.agents):
            # agents[0].actor_local.load_state_dict(torch.load('models/best/checkpoint_actor_0.pth'))
            # agents[1].actor_local.load_state_dict(torch.load('models/best/checkpoint_actor_1.pth'))
            agent.actor_local.load_state_dict(torch.load('agent{}_checkpoint_actor.pth'.format(index+1)))
            agent.critic_local.load_state_dict(torch.load('agent{}_checkpoint_critic.pth'.format(index+1)))


    def reset(self):        
        for agent in self.agents:
            agent.reset()

    def train(self, n_episodes=5000, max_t=int(1000)):
        scores_deque = deque(maxlen=100)
        scores = []
        average_scores_list = []

        for i_episode in range(1, n_episodes+1):                                    
            env_info = self.env.reset(train_mode=True)[self.brain_name]     
            states = env_info.vector_observations               
            score = np.zeros(self.num_agents)

            self.reset()

            for t in range(max_t):
                actions = self.act(states)
                env_info = self.env.step(actions)[self.brain_name]            
                next_states = env_info.vector_observations
                rewards = env_info.rewards         
                dones = env_info.local_done                         
                self.step(states, actions, rewards, next_states, dones, t)        
                states = next_states
                score += rewards  

                if any(dones):                                 
                    break

            score_max = np.max(score)
            scores.append(score_max)
            scores_deque.append(score_max)
            average_score = np.mean(scores_deque)
            average_scores_list.append(average_score)

            print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_deque)), end="")  

            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage score: {:.3f}'.format(i_episode , average_score))

            if average_score >= 0.5:
                self.save_weights()
                print("\rSolved in episode: {} \tAverage score: {:.3f}".format(i_episode , average_score))
                break
        return scores , average_scores_list

    def test(self, env, state_size, max_t=1000, train_mode=False):
        # load best performing models
        self.load_weights()
        brain_name = env.brain_names[0]

        env_info = env.reset(train_mode=train_mode)[brain_name]          # reset the environment
        states = env_info.vector_observations                            # get the states from the environment
        scores = np.zeros(self.num_agents)                               # initialise scores to zero
        for t in range(max_t):                    
            actions = self.act(states)                                   # choose agent actions and combine them
            env_info = env.step(actions)[brain_name]                     # send both agents' actions together to the environment
            next_states = env_info.vector_observations                   # get the next states from the environment
            rewards = env_info.rewards                                   # get reward
            done = env_info.local_done                                   # see if episode finished
            scores += np.max(rewards)                                    # update the score for each agent
            states = next_states                                         # roll over states to next time step
        return

# Replay Buffer Taken from First Project
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, num_agents, seed):
        """Initialize a ReplayBuffer object. 

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal buffer (deque)
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to buffer."""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """Randomly sample a batch of experiences from buffer."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states_list = [torch.from_numpy(np.vstack([e.states[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]
        actions_list = [torch.from_numpy(np.vstack([e.actions[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]
        next_states_list = [torch.from_numpy(np.vstack([e.next_states[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]            
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)        
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states_list, actions_list, rewards, next_states_list, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
