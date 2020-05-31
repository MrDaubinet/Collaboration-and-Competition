from collections import deque
import time
import numpy as np
import torch

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size

class DDPG():
    # def __init__(self,):
    #     self.sharedBuffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
    
    @staticmethod
    def train(env, agents, state_size, action_size, n_episodes=2000, max_t=int(1000), train_mode=True, solved_score=0.5, consec_episodes=100, print_every=10, add_noise=True, actor_path='actor_ckpt.pth', critic_path='critic_ckpt.pth'):
        """Deep Deterministic Policy Gradient (DDPG)

        Params
        ======
                n_episodes (int)      : maximum number of training episodes
                max_t (int)           : maximum number of timesteps per episode
                train_mode (bool)     : if 'True' set environment to training mode
                solved_score (float)  : min avg score over consecutive episodes
                consec_episodes (int) : number of consecutive episodes used to calculate score
                print_every (int)     : interval to display results
                actor_path (str)      : directory to store actor network weights
                critic_path (str)     : directory to store critic network weights
        """

        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)

        best_score = -np.inf
        scores_all = []											                        # list containing scores from each episode
        score_window = deque(maxlen=100)							                    # last 100 scores
        best_episode = 0                                                                # The best performing episode
        scores_window_avg = []										                    # average of last 100 scores
        solved = False

        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]   	                    # reset environment
            states = DDPG.get_states(agents, env_info, state_size)              # get current state for each agent
            for agent in agents:
                agent.reset()
            
            scores = np.zeros(num_agents)                           		            # initialize score for each agent
            start_time = time.time()
            # perhaps running infinite loop here
            for t in range(max_t):
                # actions = agent.act(states)								                    
                actions = DDPG.get_actions(agents, states, action_size, add_noise)      # select actions
                env_info = env.step(actions)[brain_name]                                # send actions to environment
                next_states = DDPG.get_states(agents, env_info, state_size)     # select states
                rewards = env_info.rewards							                    # get reward
                done = env_info.local_done                                              # see if episode has finished

                # save experience to replay buffer,
                DDPG.learning_step(states, actions, rewards, next_states, done)
                scores += np.max(rewards)                                               # update scores with best reward
                states = next_states
                                           	
                if np.any(done):                                       			
                    break

            duration = time.time() - start_time  						                # update time
            ep_best_score = np.max(scores)                                              # record best score for episode
            scores_window.append(ep_best_score)                                         # add score to recent scores
            scores_all.append(ep_best_score)                                            # add score to histor of all scores
            moving_average.append(np.mean(scores_window))                               # recalculate moving average
            
            # save best score
            if ep_best_score > best_score:
                best_score = ep_best_score
                best_episode = i_episode

            # print results
            if i_episode % PRINT_EVERY == 0:
                print('Episodes {:0>4d}-{:0>4d}\tEpisode Time: {:.1f}\tMax Reward: {:.3f}\tMoving Average: {:.3f}'.format(
                    i_episode-PRINT_EVERY, i_episode, duration, np.max(scores_all[-PRINT_EVERY:]), moving_average[-1]))      

            # return mean_scores, mean_scores_window_avg
            # determine if environment is solved and keep best performing models
            if moving_average[-1] >= SOLVED_SCORE:
                if not already_solved:
                    print('<-- Environment solved in {:d} episodes! \
                    \n<-- Moving Average: {:.3f} over past {:d} episodes'.format(
                        i_episode-CONSEC_EPISODES, moving_average[-1], CONSEC_EPISODES))
                    already_solved = True
                    # save weights
                    torch.save(agent_0.actor_local.state_dict(), 'models/checkpoint_actor_0.pth')
                    torch.save(agent_0.critic_local.state_dict(), 'models/checkpoint_critic_0.pth')
                    torch.save(agent_1.actor_local.state_dict(), 'models/checkpoint_actor_1.pth')
                    torch.save(agent_1.critic_local.state_dict(), 'models/checkpoint_critic_1.pth')
                elif ep_best_score >= best_score:
                    print('<-- Best episode so far!\
                    \nEpisode {:0>4d}\tMax Reward: {:.3f}\tMoving Average: {:.3f}'.format(
                    i_episode, ep_best_score, moving_average[-1]))
                    # save weights
                    torch.save(agent_0.actor_local.state_dict(), 'models/checkpoint_actor_0.pth')
                    torch.save(agent_0.critic_local.state_dict(), 'models/checkpoint_critic_0.pth')
                    torch.save(agent_1.actor_local.state_dict(), 'models/checkpoint_actor_1.pth')
                    torch.save(agent_1.critic_local.state_dict(), 'models/checkpoint_critic_1.pth')
                # stop training if model stops improving
                elif (i_episode-best_episode) >= 200:
                    print('<-- Training stopped. Best score not matched or exceeded for 200 episodes')
                    break
                else:
                    continue
            env.close()

    @staticmethod
    def test(agents, env, state_size, actor_path='actor_ckpt.pth', critic_path='critic_ckpt.pth', train_mode=False):
        # load best performing models
        agents[0].actor_local.load_state_dict(torch.load('models/best/checkpoint_actor_0.pth'))
        agents[1].actor_local.load_state_dict(torch.load('models/best/checkpoint_actor_1.pth'))

        brain_name = env.brain_names[0]
        # num_agents = len(env_info.agents)

        env_info = env.reset(train_mode=train_mode)[brain_name]                         # reset the environment
        states = np.reshape(env_info.vector_observations, (1,len(agents)*state_size))    # get states and combine them
        scores = np.zeros(len(agents))
        while True:
            actions = get_actions(states, ADD_NOISE)                                    # choose agent actions and combine them
            env_info = env.step(actions)[brain_name]                                    # send both agents' actions together to the environment
            next_states = get_states(agents, env_info, states, state_size)              # combine the agent next states
            rewards = env_info.rewards                                                  # get reward
            done = env_info.local_done                                                  # see if episode finished
            scores += np.max(rewards)                                                   # update the score for each agent
            states = next_states                                                        # roll over states to next time step
            if np.any(done):                                                            # exit loop if episode finished
                break

        return scores
    
    @staticmethod
    def get_actions(agents, states, action_size, add_noise=True):
        # actions = [agent.act(states, add_noise) for agent in agents]
        # # flatten action pairs into a single vector
        # return np.concatenate((actions[0], actions[1]), axis=0).flatten()
        action_0 = agents[0].act(states, add_noise)    # agent 0 chooses an action
        action_1 = agents[1].act(states, add_noise)    # agent 1 chooses an action
        return np.concatenate((action_0, action_1), axis=0).flatten()

    @staticmethod
    def get_states(agents, env_info, state_size):
        states = env_info.vector_observations, 
        return np.reshape(states, (1,48))

    @staticmethod
    def learning_step(states, actions, rewards, next_states, done):
        for i, agent in enumerate(agents):
            agent.step(states, actions, rewards[i], next_states, done, i)
