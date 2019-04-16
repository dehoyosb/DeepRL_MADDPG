import torch
import torch.nn.functional as F

import numpy as np

from ddpg import Agent
from buffer import ReplayBuffer

BUFFER_SIZE = int(3e5)  # replay buffer size
NOISE_START = int(1e4)
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 20
UPDATE_AMOUNT = 15

huber_loss = torch.nn.SmoothL1Loss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    def __init__(self, num_agents, state_size, action_size, random_seed):
        super(MADDPG, self).__init__()
        
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed

        self.maddpg_agent = [Agent(self.state_size, self.action_size,
                                   self.num_agents*self.state_size, self.num_agents*self.action_size,
                                   self.random_seed) for i in range(self.num_agents)]
        
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        self.noise_amplitud = 1
        self.noise_reduction = 0.9995
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step += 1
        if len(self.memory) > BATCH_SIZE and self.t_step % UPDATE_EVERY == 0:
            # Learn, if enough samples are available in memory
            for _ in range(round(UPDATE_AMOUNT)):
                for agent in range(self.num_agents):
                    experiences = self.memory.sample()
                    self.learn(experiences, agent ,GAMMA)
                self.update_targets()

    def act(self, states):
        """get actions from all agents in the MADDPG object"""
        if self.t_step < NOISE_START:
            noise_amplitud = 0
        else:
            noise_amplitud = self.noise_amplitud
            self.noise_amplitud = max(self.noise_amplitud*self.noise_reduction, 0.1)
            
        actions = np.array([agent.act(state, noise_amplitud) for agent, state in zip(self.maddpg_agent, states)])
            
        return actions

    def target_actors(self, states):
        target_actions = torch.cat([agent.actor_target(states[:,i,:]) for i, agent in enumerate(self.maddpg_agent)], dim = 1)
        return target_actions
    
    def actors(self, states):
        actions = torch.cat([agent.actor(states[:,i,:]) for i, agent in enumerate(self.maddpg_agent)], dim = 1)
        return actions
    
    def learn(self, experiences, agent_number ,gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        agent = self.maddpg_agent[agent_number]

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        target_actions_full = self.target_actors(next_states)
        next_states_full = next_states.view(-1,self.num_agents*self.state_size)
#         target_critic_input = torch.cat((next_states_full,target_actions_full), dim = 1)
        
        Q_targets_next = agent.critic_target(next_states_full,target_actions_full)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards[:,agent_number].view(-1,1) + (gamma * Q_targets_next * (1 - dones[:,agent_number].view(-1,1)))

        # Compute critic loss
        actions_full = actions.view(-1,self.action_size*self.num_agents)
        states_full = states.view(-1,self.num_agents*self.state_size)
#         critic_input = torch.cat((states_full,actions_full), dim = 1)
        
        Q_expected = agent.critic(states_full,actions_full)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
#         critic_loss = huber_loss(Q_expected, Q_targets.detach())
        
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_full_pred = self.actors(states)
#         critic_input_loss = torch.cat((states_batch, actions_full), dim = 1)
        actor_loss = -agent.critic(states_full,actions_full_pred).mean()
        
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1)
        agent.actor_optimizer.step()
        
    def update_targets(self):
        """soft update target networks"""
        for agent in self.maddpg_agent:
            self.soft_update(agent.actor, agent.actor_target, TAU)
            self.soft_update(agent.critic, agent.critic_target, TAU)
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def reset(self):
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.noise.reset()
