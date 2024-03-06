import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Actor(nn.Module):

    def __init__(self, dim_states, dim_actions, continuous_control):
        super(Actor, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_states -> 64 -> 64 -> dim_actions
        self.fc1 = nn.Linear(dim_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim_actions)
        self.relu = nn.ReLU()
        self.continuous_control = continuous_control

        if continuous_control:
            # trainable parameter
            self._log_std = nn.Parameter(torch.zeros(dim_actions))


    def forward(self, input):
        x1 = self.relu(self.fc1(input))
        x2 = self.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3


class Critic(nn.Module):

    def __init__(self, dim_states):
        super(Critic, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_states -> 64 -> 64 -> 1
        self.fc1 = nn.Linear(dim_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        x1 = self.relu(self.fc1(input))
        x2 = self.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3


class ActorCriticAgent:

    def __init__(self, dim_states, dim_actions, actor_lr, critic_lr, gamma, continuous_control=False):
        
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._gamma = gamma

        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._continuous_control = continuous_control

        self._actor = Actor(self._dim_states, self._dim_actions, self._continuous_control)

        # Adam optimizer
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=self._actor_lr)

        self._critic = Critic(self._dim_states)

        # Adam optimizer
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=self._critic_lr)

        self._select_action = self._select_action_continuous if self._continuous_control else self._select_action_discrete
        self._compute_actor_loss = self._compute_actor_loss_continuous if self._continuous_control else self._compute_actor_loss_discrete


    def select_action(self, observation):
        return self._select_action(observation)
        

    def _select_action_discrete(self, observation):
        # sample from categorical distribution
        with torch.no_grad():
            obs_tensor = torch.tensor(observation)
            out = self._actor(obs_tensor)
            categorical = torch.distributions.Categorical(logits=out)
            action = categorical.sample().item()
        return action

    def _select_action_continuous(self, observation):
        # sample from normal distribution
        # use the log std trainable parameter
        with torch.no_grad():
            obs_tensor = torch.tensor(observation)
            mean = self._actor(obs_tensor)
            std = torch.exp(self._actor._log_std)
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
        return action.numpy()


    def _compute_actor_loss_discrete(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        obs_batch_tensor = torch.tensor(observation_batch)
        action_batch_tensor = torch.tensor(action_batch)
        advantage_batch_tensor = torch.tensor(advantage_batch)

        logits = self._actor(obs_batch_tensor)
        categorical = torch.distributions.Categorical(logits=logits)
        log_prob = categorical.log_prob(action_batch_tensor)
        loss = -(log_prob*advantage_batch_tensor).mean()
        return loss


    def _compute_actor_loss_continuous(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        obs_batch_tensor = torch.tensor(observation_batch)
        action_batch_tensor = torch.tensor(action_batch)
        advantage_batch_tensor = torch.tensor(advantage_batch)

        mean = self._actor(obs_batch_tensor)
        std = torch.exp(self._actor._log_std)
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action_batch_tensor).squeeze()
        loss = -(log_prob*advantage_batch_tensor).mean()
        return loss


    def _compute_critic_loss(self, observation_batch, reward_batch, next_observation_batch, done_batch):
        # minimize mean((r + gamma * V(s_t1) - V(s_t))^2)
        obs_batch_tensor = torch.tensor(observation_batch)
        reward_batch_tensor = torch.tensor(reward_batch)
        next_obs_batch_tensor = torch.tensor(next_observation_batch)
        mask_tensor = torch.tensor(1-done_batch)

        V_t = self._critic(obs_batch_tensor).squeeze().float()
        
        with torch.no_grad(): 
            V_t1 = self._critic(next_obs_batch_tensor).squeeze()
            V_target = reward_batch_tensor + (mask_tensor*V_t1*self._gamma)
        
        loss = F.mse_loss(V_t, V_target.float())
        return loss


    def update_actor(self, observation_batch, action_batch, reward_batch, next_observation_batch, done_batch):
        # compute the advantages using the critic and update the actor parameters
        # use self._compute_actor_loss
        advantage_batch = self._compute_advantage_batch_tensor(observation_batch, reward_batch, next_observation_batch, done_batch)
        loss = self._compute_actor_loss(observation_batch, action_batch, advantage_batch)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return loss.item()        
        
    def update_critic(self, observation_batch, reward_batch, next_observation_batch, done_batch):
        # update the critic
        # use self._compute_critic_loss
        loss = self._compute_critic_loss(observation_batch, reward_batch, next_observation_batch, done_batch)
        self._critic_optimizer.zero_grad()
        loss.backward()
        self._critic_optimizer.step()
        return loss.item()
    
    def _compute_advantage_batch_tensor(self, observation_batch, reward_batch, next_observation_batch, done_batch):
        obs_batch_tensor = torch.tensor(observation_batch)
        reward_batch_tensor = torch.tensor(reward_batch)
        next_obs_batch_tensor = torch.tensor(next_observation_batch)
        mask_tensor = torch.tensor(1-done_batch)

        V_t1 = self._critic(next_obs_batch_tensor).squeeze()
        V_t = self._critic(obs_batch_tensor).squeeze()

        target_value_tensor = reward_batch_tensor + self._gamma*V_t1*mask_tensor
        advantage_tensor = target_value_tensor - V_t

        return advantage_tensor.detach().numpy()