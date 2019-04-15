import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from noise import OUNoise
from buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

LR_ACTOR = 5e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

class Agent():
    def __init__(self, state_size, action_size, state_size_full, action_size_full, random_seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic = Critic(state_size_full, action_size_full, random_seed).to(device)
        self.critic_target = Critic(state_size_full, action_size_full, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # initialize targets same as original networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        
    def hard_update(self, target, source):
        """
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def act(self, state, amplitud):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.squeeze().numpy()
        self.actor.train()
        action += amplitud*self.noise.sample()
        return np.clip(action, -1, 1)