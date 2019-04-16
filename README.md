# MADDPG
Multi Agent Deep Deterministic Policy Gradient Implementation

In this repository it's shown the usage of a Deep Reinforcement Tecnique called DDPG or Deep Deterministic Policy Gradient, which uses the concepts of Reinforcement Learning applied in a controlled enviroment. Using a Deep Neural Network as a Function Approximator to map state to action pairs, in order to decide how an agent has to act given it's current state and the discounted cummulative reward. Additionally, this tecnique is able to manage any enviroment that has a set of continuos actions, which makes it extremaly useful in real life applications. For this particular case, the training is done with 2 agents which are learning simultaniously through self-play.

## The Enviroment
The objective of the enviroment is to be used as a playground to train two agents that are capable of playing tennis with one another. The agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically, after each episode the rewards that each agent received (without discounting) are added up , to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode. The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Setting Up the Enviroment
Follow the instructions below to explore the environment on your own machine! You will also learn how to use the Python API to control your agent.

### Step 1: Clone the DRLND Repository
Please follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

If you are having troubles connecting to the enviroment's jupyter notebook kernel, remember to reinstall jupyter on that particular enviroment. If you are using Linux, use the next line of code on your terminal: ```conda install jupyter```

### Step 2: Download the Unity Environment
For this project, you will not need to install Unity - this is because there is an already built environment to work with, and you can download it from one of the links below. You need only select the environment that matches your operating system:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the ```p3_collab-compet``` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

## Running the Agent
Once you have set up the enviroment, go to **5. Testing Agent After Learning** in Tennis.ipynb, and run all the cells except the last one which closes the enviroment. If the path to your enviroment is different, just change it in the cell below **Loading Enviroment**. You will be able to see in the unity window how the agent is sticking to its target position, which is moving in a continuos manner, as the agent's own joints. If you want to see various episode runs, just rerun the times you want the cell below **Running Episode**
In the Case you want to train the agent from scratch, run all the cells from **4. It's Your Turn** section, and you'll be able to see the training process and the reward plot per 100 episodes. There you will save the weights learned, and the files can be used to test the agent thereafter.


