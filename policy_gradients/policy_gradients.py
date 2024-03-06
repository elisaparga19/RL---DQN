import torch 
import torch.nn as nn

import numpy as np

class Policy(nn.Module):

    def __init__(self, dim_states, dim_actions, continuous_control):
        super(Policy, self).__init__()
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


class PolicyGradients:

    def __init__(self, dim_states, dim_actions, lr, gamma, 
                 continuous_control=False, reward_to_go=False, use_baseline=False):
        
        self._learning_rate = lr
        self._gamma = gamma
        
        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._continuous_control = continuous_control
        self._use_reward_to_go = reward_to_go
        self._use_baseline = use_baseline

        self._policy = Policy(self._dim_states, self._dim_actions, self._continuous_control)
        # Adam optimizer
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=self._learning_rate)

        self._select_action = self._select_action_continuous if self._continuous_control else self._select_action_discrete
        self._compute_loss = self._compute_loss_continuous if self._continuous_control else self._compute_loss_discrete


    def select_action(self, observation):
        return self._select_action(observation)   

    def _select_action_discrete(self, observation):
        # sample from categorical distribution
        with torch.no_grad():
            obs_tensor = torch.tensor(observation)
            out = self._policy(obs_tensor)
            categorical = torch.distributions.Categorical(logits=out)
            action = categorical.sample().item()
        return action

    def _select_action_continuous(self, observation):
        # sample from normal distribution
        # use the log std trainable parameter
        with torch.no_grad():
            obs_tensor = torch.tensor(observation)
            mean = self._policy(obs_tensor)
            std = torch.exp(self._policy._log_std)
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
        return action.numpy()            

    def update(self, observation_batch, action_batch, advantage_batch):
        # update the policy here
        # you should use self._compute_loss
        loss = -self._compute_loss(observation_batch, action_batch, advantage_batch) # largo k
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()
    

    def _compute_loss_discrete(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        obs_batch_tensor = torch.tensor(observation_batch)
        action_batch_tensor = torch.tensor(action_batch)
        advantage_batch_tensor = torch.tensor(advantage_batch)

        logits = self._policy(obs_batch_tensor)
        categorical = torch.distributions.Categorical(logits=logits)
        log_prob = categorical.log_prob(action_batch_tensor) # squeeze
        loss = (log_prob*advantage_batch_tensor).mean()
        return loss


    def _compute_loss_continuous(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        obs_batch_tensor = torch.tensor(observation_batch)
        action_batch_tensor = torch.tensor(action_batch)
        advantage_batch_tensor = torch.tensor(advantage_batch)

        mean = self._policy(obs_batch_tensor)
        std = torch.exp(self._policy._log_std)
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action_batch_tensor).squeeze()
        loss = (log_prob*advantage_batch_tensor).mean()
        return loss

    
    def estimate_returns(self, rollouts_rew):
        estimated_returns = []
        for rollout_rew in rollouts_rew:

            if self._use_reward_to_go:
                # only for part 2
                estimated_return = self._reward_to_go(rollout_rew)
            else:
                estimated_return = self._discount_rewards(rollout_rew)
            
            estimated_returns = np.concatenate([estimated_returns, estimated_return])

        if self._use_baseline:
            # only for part 2
            average_return_baseline = estimated_returns.mean()
            # Use the baseline:
            estimated_returns -= average_return_baseline

        return np.array(estimated_returns, dtype=np.float32)


    # It may be useful to discount the rewards using an auxiliary function [optional]
    def _discount_rewards(self, rewards):
        T = len(rewards)
        cum_rewards = 0
        for i in range(T):
            cum_rewards += (self._gamma**i)*rewards[i]
        discount_rewards = [cum_rewards] * T
        return np.array(discount_rewards)

    def _reward_to_go(self, rewards):
        rew_to_go = np.zeros_like(rewards)
        cum_rewards = 0
        for i in reversed(range(len(rewards))):
            cum_rewards = cum_rewards * self._gamma + rewards[i]
            rew_to_go[i] =  cum_rewards
        return rew_to_go
    
