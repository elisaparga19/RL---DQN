import torch
import torch.nn as nn

import copy

import numpy as np


class DeepQNetwork(nn.Module):

    def __init__(self, dim_states, dim_actions):
        super(DeepQNetwork, self).__init__()
        self._fc1 = nn.Linear(dim_states, 64)
        self._fc2 = nn.Linear(64, 64)
        self._fc3 = nn.Linear(64, dim_actions)
        self._relu = nn.functional.relu

    def forward(self, input):
        output = self._relu(self._fc1(input))
        output = self._relu(self._fc2(output))
        output = self._fc3(output)
        return output
    

class ConservativeDeepQNetworkAgent:

    def __init__(self, dim_states, dim_actions, lr, gamma, alpha):
        
        self._learning_rate = lr
        self._gamma = gamma
        self._alpha = alpha

        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._deep_qnetwork = DeepQNetwork(self._dim_states, self._dim_actions)
        self._target_deepq_network = copy.deepcopy(self._deep_qnetwork)

        self._optimizer = torch.optim.Adam(self._deep_qnetwork.parameters(), lr=self._learning_rate)
        self._loss = nn.MSELoss()


    def replace_target_network(self):
        with torch.no_grad():
            for param, target_param in zip(self._deep_qnetwork.parameters(), 
                                           self._target_deepq_network.parameters()):
                target_param.data = param


    def select_action(self, observation, greedy=False):

        # Select action greedily
        with torch.no_grad():
            action = self._deep_qnetwork(torch.from_numpy(np.array(observation, dtype=np.float32))).argmax().numpy()

        return action


    def update(self, experiences_batch):

        s_t_batch, a_t_batch, r_t_batch, s_t1_batch, done_t_batch = experiences_batch

        # To tensor:
        s_t_batch = torch.tensor(s_t_batch).float()
        s_t1_batch = torch.tensor(s_t1_batch).float()
        a_t_batch = torch.tensor(a_t_batch).unsqueeze(1).long()
        r_t_batch = torch.tensor(r_t_batch).float()
        done_t_batch = torch.tensor(done_t_batch).float()

        # P2 - 1
        with torch.no_grad():
            target = self._target_deepq_network(s_t1_batch)
            max_q_target, _ = torch.max(target, dim=1)
            target = r_t_batch + self._gamma*max_q_target*(1-done_t_batch)


        q_val = self._deep_qnetwork(s_t_batch)
        q_val_exp = q_val.gather(1, a_t_batch).squeeze(1)

        self._optimizer.zero_grad()

        dqn_loss = self._loss(q_val_exp, target)
        # CQL1 loss (check the paper)
        # hint: use torch.logsumexp
        c_target = torch.logsumexp(q_val, dim=1).mean() - q_val_exp.mean()

        loss = dqn_loss + self._alpha*c_target # DQN loss + self._alpha * c_target

        loss.backward()

        self._optimizer.step()
