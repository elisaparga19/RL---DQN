import torch
import torch.nn as nn

import copy

import numpy as np

from replay_buffer import ReplayBuffer


class DeepQNetwork(nn.Module):

    def __init__(self, dim_states, dim_actions):
        super(DeepQNetwork, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_states -> 64 -> 64 -> dim_actions
        self.fc1 = nn.Linear(dim_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim_actions)
        self.relu = nn.ReLU()

    def forward(self, input):
        x1 = self.relu(self.fc1(input))
        #x1 = self.relu(x1)
        x2 = self.relu(self.fc2(x1))
        #x2 = self.relu(x2)
        x3 = self.fc3(x2)
        return x3


class DeepQNetworkAgent:

    def __init__(self, dim_states, dim_actions, lr, gamma, epsilon, nb_training_steps, replay_buffer_size, batch_size):
        
        self._learning_rate = lr
        self._gamma = gamma
        self._epsilon = epsilon
        self._nb_training_steps = nb_training_steps

        self._epsilon_min = 0
        self._epsilon_decay = self._epsilon / (self._nb_training_steps / 2.)

        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self.replay_buffer = ReplayBuffer(dim_states=self._dim_states,
                                          dim_actions=self._dim_actions,
                                          max_size=replay_buffer_size,
                                          sample_size=batch_size)

        # Complete
        self._deep_qnetwork = DeepQNetwork(self._dim_states, self._dim_actions)
        self._target_deepq_network = copy.deepcopy(self._deep_qnetwork)

        # Adam optimizer
        self._optimizer = torch.optim.Adam(self._deep_qnetwork.parameters(), lr=self._learning_rate)

        # MSE Loss
        self._loss = nn.MSELoss()


    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):
        self.replay_buffer.store_transition(s_t, a_t, r_t, s_t1, done_t)


    def replace_target_network(self):
        # Esta funcion debe copiar los parametros de Q-Network en Target Q-Network.
        self._target_deepq_network.load_state_dict(self._deep_qnetwork.state_dict())


    def select_action(self, observation, greedy=False):

        if np.random.random() > self._epsilon or greedy:
            # Select action greedily
            with torch.no_grad():
                actions_out = self._deep_qnetwork(torch.tensor(observation))
                action = torch.argmax(actions_out)

        else:
            # Select random action
            action = torch.randint(0, int(self._dim_actions), (1,)).squeeze()

        if not greedy and self._epsilon >= self._epsilon_min:
            # Implement epsilon linear decay
            self._epsilon -= self._epsilon_decay
               
        return action.item()


    def update(self):
        
        sample = self.replay_buffer.sample_transitions()
        
        sample_st = torch.tensor(sample[0]).float()
        sample_at = torch.tensor(sample[1]).float()
        sample_rt = torch.tensor(sample[2]).float()
        sample_st1 = torch.tensor(sample[3]).float()
        sample_done = torch.tensor(sample[4]).float()

        q_val = self._deep_qnetwork(sample_st)
        q_val = q_val.gather(1, sample_at.long().unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_target = self._target_deepq_network(sample_st1)
            max_q_target, _ = torch.max(q_target, axis=1)
            y_j = sample_rt + (1-sample_done)*self._gamma*max_q_target
                  
        loss = self._loss(q_val, y_j)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()
