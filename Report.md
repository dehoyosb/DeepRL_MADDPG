## Report

### The Enviroment
The objective of the enviroment is to be used as a playground to train two agents that are capable of playing tennis with one another. The agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically, after each episode the rewards that each agent received (without discounting) are added up , to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode. The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### The Learning Algorithm
To solve the enviroment i used the Deep Deterministic Policy Gradient Algorithm (DDPG) but with the multiagent factor. This is an Actor- Critic method, that means that we have a neural network (The Actor) that interacts with the enviroment given the input states. The Actor for this method is deterministic, that means that we are not recieving a probability distribution over the actions, rather they are being approximated by the neural network directly. Afterwards, the actions approximated by the Actor are fed back as input to the Critic (Q(s, a), substitute the actor network for a), which is another neural network that approximates the action-value functions and its trying to maximize them to find the optimal policy. This is accomplished by expressing the output of the critic in terms of loss of the actor, i.e. loss(actor) = - output(critic). We backpropagate that loss to update the Actor Networks.
Now, since this problem has a multi agent characteristic, we have and actor and a critic per agent. Both actors are recieving the agents own observations of the environment and are giving their predicted outputs. But the difference here, is that each critic is recieving a full observation of both agents states and both agents actions.
Given that the learning is off-policy, we need a replay buffer to store experiences from the agent. Therefore as in the DQN algorithm we have a local and a target network as well, one that interacts with the enviroment and the other one that is being updated constantly.We have then 2 netowrks for the Actors and 2 Networks for the Critics. After a few steps, we make a soft update to the networks from the experiences stored away. Learning from the Critic is done with Q learning, where we are updating the Q values, from the predicted  next Q value from next state, times the discounting factor, plus the reward. Then, the loss of the critic, loss(critic) = F.mse_loss(Q_expected, Q_targets). Where Q_expected comes from the output of the local Critic Network, and Q_targets comes from the adjusted Q learning step. Finally we backpropagate that loss to update the Critic Networks. 

In summary, the algorithm has 2 important processes that work inside it. In the first one, the agents sample the enviroment by performing actions and stores away the different experience tuples in a replay memory. In the second one, the agents select a random batch of experiences from the replay buffer and learns from it using a gradient descent step. Iterating over these 2 steps leads the agent to learn how to interact with the enviroment abstracting the optimal policy from the learned action value function. And given the full observations input to the critic networks of both agents,the agents are learning to cooperate to keep the tennis ball as long as posible on the field.

### Neural Networks
The Actor Neural networks are a 3 layer feed forward network with Relu activation between layers and Tanh in the output layer, given that the actions have to be sampled from a continous space of -1 to 1. The output of the actor networks have the size of the enviroment action space, that is a vector of 2. 

The Critic Neural networks are a 3 layer feed forward network with Relu activation between layers. In the first layer, the network is fed with the full state observations of both agents of the enviroment; in the second layer, the output from the first layer is concatenated with the full action space given by both agent's actor networks. Finally, the critic networks outputs are a single value with the approximated value function given the actors actions.

Given that for this particular problem the input are not the raw pixels from the enviroment, rather the 24 states per agent that contain the position and velocity of the ball and racket, there was not need for the implementation of a convolutional neural network.

### The Hyperparameters
- Buffer Size = 300000
- Batch Size = 256
- Gamma = 0.99
- Tau = 0.001
- Actor Learning Rate = 0.0001
- Critic Learning Rate = 0.001

The Actor and Critic networks were updated through the learning process, every 20 time steps if the memory buffer had at least the same ammount of experiences as the minimum batch size. Every 20 steps a 15 train loop was executed to maximize learning.

- Number of Episodes = 1000
- Max number of Steps per Episode = 1000

Also the Ornstein-Uhlenbeck process which adds noise to the actions selected by the Actor Nework, a normal distributions was used instead of a uniform distribution. OUnoise was applied with an amplitud scale decreasing over time.

- starting_noise_amplitud = 1
- noise_reduction_per_step = 0.9995
- minimum_noise_amplitud = 0.1

This was done to maximize exploration at the early stages of experiences and learning, but at the end we would just have a minimal ammount of exploration.

### Plot of Rewards per Episode
![Image of Reward PLot](/training_results/reward_plot_maddpg.png)
![Image of Training Process](/training_results/trainin_process_maddpg.png)

The Enviroment was solved around episode 1400

### Ideas for Improving Agents Performance
One of the things that may be hindering performance is the random selection of experiences from the memory buffer. Why? because not every experience is the best experience that can be used to learn. There are some specific experiences that should be used more than others to improve learning in some aspects. Also, there are some experiences that are less common than others, and some of those may have valuable information that can not be neglected. So, if one could give a weight to each particular experience in a way that allows to take one particular experience more often than others, learning could be improved and could actually converge faster.
