### Import necessary libraries


```python
import numpy as np
import gym
```

### Create an environment and get its size.


```python
env = gym.make("FrozenLake-v0")
n_observations = env.observation_space.n
n_actions = env.action_space.n
```

    [2023-01-14 14:20:47,407] Making new env: FrozenLake-v0
    

### Initialize the Q-table to 0


```python
Q_table = np.zeros((n_observations,n_actions))
print(Q_table)
```

    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    

### Initialize Hyper Parameters


```python
#number of episode we will run
n_episodes = 10000

#maximum of iteration per episode
max_iter_episode = 100

#initialize the exploration probability to 1
exploration_proba = 1

#exploartion decreasing decay for exponential decreasing
exploration_decreasing_decay = 0.001

# minimum of exploration proba
min_exploration_proba = 0.01

#discounted factor
gamma = 0.99

#learning rate
lr = 0.1
```

### Declare a list to store rewards we get after each episode


```python
rewards_per_episode = list()
```

### Main Loop to iterate over episodes to calculate rewards and performance


```python
#we iterate over episodes
for e in range(n_episodes):
    #we initialize the first state of the episode
    current_state = env.reset()
    done = False
    
    #sum the rewards that the agent gets from the environment
    total_episode_reward = 0
    
    for i in range(max_iter_episode): 
        # we sample a float from a uniform distribution over 0 and 1
        # if the sampled flaot is less than the exploration proba
        #     the agent selects arandom action
        # else
        #     he exploits his knowledge using the bellman equation 
        
        if np.random.uniform(0,1) < exploration_proba:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[current_state,:])
        
        # The environment runs the chosen action and returns
        # the next state, a reward and true if the epiosed is ended.
        next_state, reward, done, _ = env.step(action)
        
        # We update our Q-table using the Q-learning iteration
        Q_table[current_state, action] = (1-lr) * Q_table[current_state, action] +lr*(reward + gamma*max(Q_table[next_state,:]))
        total_episode_reward = total_episode_reward + reward
        # If the episode is finished, we leave the for loop
        if done:
            break
        current_state = next_state
    #We update the exploration proba using exponential decay formula 
    exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
    rewards_per_episode.append(total_episode_reward)
```

### Print all rewards and observe, with each iteration, the performance is increasing.


```python
print("Mean reward per thousand episodes")
for i in range(10):
    print((i+1)*1000,": mean espiode reward: ", np.mean(rewards_per_episode[1000*i:1000*(i+1)]))
```

    Mean reward per thousand episodes
    1000 : mean espiode reward:  0.047
    2000 : mean espiode reward:  0.205
    3000 : mean espiode reward:  0.445
    4000 : mean espiode reward:  0.559
    5000 : mean espiode reward:  0.665
    6000 : mean espiode reward:  0.71
    7000 : mean espiode reward:  0.685
    8000 : mean espiode reward:  0.71
    9000 : mean espiode reward:  0.668
    10000 : mean espiode reward:  0.676
    

## Inference

1. The performance is bad at the beginning but keeps improving with each iteration.
2. Q-learning algorithm is a very efficient way for an agent to learn how the environment works.
3. The agent would also need many more episodes to learn about the environment and deliver as per the expected performance.
4. As a solution, we can use a Deep Neural Network (DNN) to approximate the Q-Value function since DNNs are known for their efficiency to approximate functions.


```python

```
