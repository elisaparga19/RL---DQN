import torch
import torch.nn as nn
import numpy as np

from buffer import Buffer

class Model(nn.Module):

    def __init__(self, dim_states, dim_actions, continuous_control):
        super(Model, self).__init__()

        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_input -> 64 -> 64 -> dim_states
        if continuous_control:
            self.fc1 = nn.Linear(dim_states + dim_actions, 64)
        else:
            self.fc1 = nn.Linear(dim_states + 1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim_states)
        self.relu = nn.ReLU()
        self.continuous_control = continuous_control


    def forward(self, state, action):
        x = torch.cat([state, action], dim=1).float()
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class RSPlanner:

    def __init__(self, dim_states, dim_actions, continuous_control, model, planning_horizon, nb_trajectories, reward_function):
        self._dim_states = dim_states
        self._dim_actions = dim_actions
        self._continuous_control = continuous_control

        self._model = model

        self._planning_horizon = planning_horizon
        self._nb_trajectories = nb_trajectories
        self._reward_function = reward_function

        
    def generate_plan(self, observation):
        # Generate a sequence of random actions
        if self._continuous_control:
            random_actions = np.random.uniform(-2, 2, size=(self._nb_trajectories, self._planning_horizon))
        else:
            random_actions = np.random.choice([0, 1], size=(self._nb_trajectories, self._planning_horizon))
        
        # Construct initial observation 
        o_t = observation
        o_t = torch.tensor(o_t).expand(self._nb_trajectories, self._dim_states).float()

        rewards = torch.zeros((self._nb_trajectories, ))
        for i in range(self._planning_horizon):
            # Get a_t
            a_t = torch.tensor(random_actions[:, i]).view(-1, 1)
            
            # Predict next observation using the model
            with torch.no_grad():
                o_t1 = self._model(o_t, a_t)

            # Compute reward (use reward_function)
            rewards += self._reward_function(o_t, a_t)
            o_t = o_t1
        
        best_plan_idx = torch.argmax(rewards)
        # Return the best sequence of actions
        return random_actions[best_plan_idx]

class MBRLAgent:

    def __init__(self, dim_states, dim_actions, continuous_control, model_lr, buffer_size, batch_size, 
                       planning_horizon, nb_trajectories, reward_function):

        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._continuous_control = continuous_control

        self._model_lr = model_lr

        self._model = Model(self._dim_states, self._dim_actions, self._continuous_control)

        # Adam optimizer
        self._model_optimizer = torch.optim.Adam(self._model.parameters(), lr = self._model_lr)

        self._buffer = Buffer(self._dim_states, self._dim_actions, buffer_size, batch_size)
        
        self._planner = RSPlanner(self._dim_states, self._dim_actions, self._continuous_control, 
                                  self._model, planning_horizon, nb_trajectories, reward_function)

        self._loss = nn.MSELoss()


    def select_action(self, observation, random=False):

        if random:
            # Return random action
            if self._continuous_control:
                return np.random.uniform(-2, 2, size=1)
            return np.random.choice([0, 1])

        # Generate plan
        plan = self._planner.generate_plan(observation)

        # Return the first action of the plan
        if self._continuous_control:
            return [plan[0]]  
        return plan[0]


    def store_transition(self, s_t, a_t, s_t1):
        self._buffer.store_transition(s_t, a_t, s_t1)


    def update_model(self):
        batches = self._buffer.get_batches()
        epoch_loss = 0
        for batch in batches:
            # Use the batches to train the model
            # loss: avg((s_t1 - model(s_t, a_t))^2)
            s_t = torch.tensor(batch[0]).float()
            a_t = torch.tensor(batch[1]).unsqueeze(1).float()
            s_t1 = torch.tensor(batch[2]).float()

            self._model_optimizer.zero_grad()

            s_pred = self._model(s_t, a_t)
            loss = self._loss(s_pred, s_t1)
            epoch_loss += loss.item()
            loss.backward()

            self._model_optimizer.step()
        return epoch_loss/len(batches)
        
        