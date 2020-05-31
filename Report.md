[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://3.bp.blogspot.com/-I6UKhtpt-pI/WzP8ThUgMRI/AAAAAAAADFQ/mmbmu0YtDeAGT1RJj0pDPPm_jYyyYYg0gCLcBGAs/s1600/image8.gif "Robots"
[image3]: https://user-images.githubusercontent.com/10624937/42135610-c37e0292-7d12-11e8-8228-4d3585f8c026.gif "Pendulum"
[image4]:  "Graph"
[image5]:  "Agent"

# Project 2: Continuous Control
## Train an agent to hold a set of balls

![artificial robots][image1]


# Background
Deep Reinforcement Learning is a trending new algorithmic approach to solve complex learning tasks. The purpose of this project is to train an agent to solve real world "Robotic Control Systems" problems like the one below:
![real robots][image2]

But in an artificial environment.


# The Environment

This project is based off of the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


Udacity created two environment options for this project, one with a single arm and another with twenty arms. After a fair number of attempts trying to master the first environment and alllllllllot! of hyperparameter optimization. I decided to implement the second environment. The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).

Specifically,
 * After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  * 
 * This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is over all 20 agents).

# Methodology

The Experimental setup is as follows:

  1. Setup the Environment.
  2. Establish a baseline with a random walk.
  3. Implement a reinforcement learning algorithm.
  4. Display Results.
  5. Conclusion of Results.
  6. Ideas for future work.


## 1. Environment Setup

Download the environment from one of the links below. You need only select the environment that matches your operating system:
  * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
  * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
  * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
  * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Import the Unity environment and create an env object
```python
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name='location of reacher.exe')
```
Info about the environment is printed out through the ```Info()``` class found [here]()  as seen below:
```
Unity Academy name: Academy
Number of Brains: 1
Number of External Brains : 1
Lesson number : 0
Reset Parameters :
  goal_speed -> 1.0
  goal_size -> 5.0

Unity brain name: ReacherBrain
Number of Visual Observations (per agent): 0
Vector Observation space type: continuous
Vector Observation space size (per agent): 33
Number of stacked Vector Observation: 1
Vector Action space type: continuous
Vector Action space size (per agent): 4
Vector Action descriptions: , , ,
created Info
Number of agents: 1
Number of actions: 4

States look like: [ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
  1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  5.75471878e+00 -1.00000000e+00
  5.55726671e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
 -1.68164849e-01]
States have length: 33
```

## 2. Establish a baseline

To evaluate the difficulty of the environment. A random walk was scored before any algorithmic implementation of the reinforcement learning agent was made. This was done by making an agent that selects random actions at each time step.
```python
env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
```
This resulted in a score of 0.03. This is quite poor when you keep in mind that the implemented agent needs to get a score of 30, consitently for 100 episodes.

## 3. Implemented Algorithm
Due to the nature of the environment being a continuous control problem. The reinforcemenr learning agorithm needs to be able to work in a continuous space. This hard requirement means we have to use a deep learning approach where neural networks are used for continuous function approximation. When considering between Policy-based vs Value-based Methods. Policy-based methods are better suited for continuous action spaces. Udacity suggest using either the [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf) or [D4PG](https://openreview.net/pdf?id=SyZipzbCb). I chose to implement the [Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1509.02971.pdf), which is describes as an extension of Deep Q-learning to continuous tasks.

I based my code off of [this](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) repository, an implementation of DDPG with OpenAI Gym's Pendulum environment. 

![Pendulum][image3]

I copied the Actor and Critic models, as [found here](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py), but I adapted the number of hidden unites to 256 and added another layer of batch normalization. I copied the agent code, [found here](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py), then changed it to accomidate 20 environments. 

The ```Agent()``` code can be [found here]() resulting DDPG algorithm can be seen below:
```python
def train(env, agent, n_episodes=500, max_t=int(1000), train_mode=True, solved_score=30.0, consec_episodes=100, print_every=1, actor_path='actor_ckpt.pth', critic_path='critic_ckpt.pth'):
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
        # get env information
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        # set the best score to negative infinity
        best_score = -np.inf
        # list containing scores from each episode
        mean_scores = []
        # last 100 scores											      
        mean_score_window = deque(maxlen=100)
        # average of last 100 scores							  
        mean_scores_window_avg = []										  
        # for each episode
        for i_episode in range(1, n_episodes+1):
            # reset environment
            env_info = env.reset(train_mode=True)[brain_name]   
            # get current state for each agent	      
            states = env_info.vector_observations                         
            num_agents = len(env_info.agents)
            agent.reset()
            # initialize score for each agent
            scores = np.zeros(num_agents)                           		  
            start_time = time.time()
            # for each iteration
            for t in range(max_t):
                # select an action
                actions = agent.act(states)	
                # send actions to environment							  
                env_info = env.step(actions)[brain_name]
                # get next state                   
                next_states = env_info.vector_observations
                # get reward                
                rewards = env_info.rewards
                # check if episode has finished							      
                dones = env_info.local_done                               
                # save experience to replay buffer,
                for state, action, reward, next_state, done in \
                  zip(states, actions, rewards, next_states, dones):
                    # Step through the experiences given by each environment
                    agent.step(state, action, reward, next_state, done, t)  
                # set the next states
                states = next_states
                # save the agents performance
                scores += rewards    
                # break if any environment is completed                            	
                if np.any(dones):                                       			
                    break
            # if the average score of the last 100 episodes, over all environments, is greater than the goal 
            # and 100 episode
            if mean_scores_window_avg[-1] >= solved_score and i_episode >= consec_episodes:
              if train_mode:
                    torch.save(agent.actor_local.state_dict(), actor_path)
                    torch.save(agent.critic_local.state_dict(), critic_path)  
                break

```

## 4. Results
My algorithm was able to solve the environment in 23 episodes with an average of 31.8 over the first 100 episodes. Check the graph below to see how it trained.
![real robots][image4]
```
Episode 90      Duration: 290.0         Mean Episode Scores: 38.52      Mean Consecutive Scores: 31.10
Episode 91      Duration: 290.0         Mean Episode Scores: 38.89      Mean Consecutive Scores: 31.19
Episode 92      Duration: 289.0         Mean Episode Scores: 39.22      Mean Consecutive Scores: 31.27
Episode 93      Duration: 289.0         Mean Episode Scores: 38.32      Mean Consecutive Scores: 31.35
Episode 94      Duration: 291.0         Mean Episode Scores: 35.74      Mean Consecutive Scores: 31.39
Episode 95      Duration: 287.0         Mean Episode Scores: 38.17      Mean Consecutive Scores: 31.47
Episode 96      Duration: 292.0         Mean Episode Scores: 39.21      Mean Consecutive Scores: 31.55
Episode 97      Duration: 294.0         Mean Episode Scores: 38.78      Mean Consecutive Scores: 31.62
Episode 98      Duration: 296.0         Mean Episode Scores: 38.03      Mean Consecutive Scores: 31.69
Episode 99      Duration: 301.0         Mean Episode Scores: 39.09      Mean Consecutive Scores: 31.76
Episode 100     Duration: 301.0         Mean Episode Scores: 38.99      Mean Consecutive Scores: 31.83
```

```
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label='DDPG')
plt.plot(np.arange(len(scores)), avgs, c='r', label='moving avg')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend(loc='upper left');
plt.show()
```
![Result Agent][image5]

## 5. Conclusion


## 6. Ideas for Future Work
* **Hyperparameter optimization** - Most algorithms can be tweeked to perform better for specific environments when by changeing the various hyper parameters. This could be investigated to find a more effective agent.
* **Priority Experience Replay** - Prioritized experience replay selects experiences based on a priority value that is correlated with the magnitude of error. This replaces the random selection of experiences with an approach that is more intelligent, as described in [this paper](https://arxiv.org/pdf/1511.05952.pdf). 