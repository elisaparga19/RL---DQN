import numpy as np

np.random.seed(1234)

class QLearningAgent():

    def __init__(self, states_high_bound, states_low_bound, nb_actions, nb_episodes, gamma, alpha, epsilon):
    
        self._epsilon = epsilon
        self._gamma = gamma
        self._alpha = alpha

        self._states_high_bound = states_high_bound
        self._states_low_bound = states_low_bound
        self._nb_actions = nb_actions
        self._nb_episodes = nb_episodes
        
        # Define these variables (P2-2)
        self._nb_grid = 30
        self._nb_states = (self._nb_grid)**2

        # 3D-array to represent Q value dor each pair state-action
        # Zero-Initialization
        self._tabular_q = np.zeros(shape=(self._nb_grid, self._nb_grid, self._nb_actions))

        # Random-Initialization
        # self._tabular_q = np.random.uniform(low = -1, high = 1, size=(self._nb_grid, self._nb_grid, self._nb_actions)) 
        
        # Grid values that generate bins to define a state
        pos_low = self._states_low_bound[0]
        pos_high = self._states_high_bound[0]
        pos_range = pos_high-pos_low
        pos_step = pos_range/self._nb_grid

        vel_low = self._states_low_bound[1]
        vel_high = self._states_high_bound[1]
        vel_range = vel_high-vel_low
        vel_step = vel_range/self._nb_grid
        
        self.position_grid = np.arange(pos_low, pos_high + pos_step, pos_step)
        self.velocity_grid = np.arange(vel_low, vel_high + vel_step, vel_step)
        

    """ Epsilon-greedy policy 
    """
    def select_action(self, observation, greedy=False):
        # P1-3
        # use this values to check in which state the agent is
        x, y = self.get_state_coordinates(observation)

        if np.random.random() > self._epsilon or greedy:
            # select action with argmax
            action = np.argmax(self._tabular_q[x,y])

        else:
            # select random action
            action = np.random.randint(0, self._nb_actions)

        return action


    """ Q-function update
    """
    def update(self, ob_t, ob_t1, action, reward, is_done):
        # P1-3
        terminal_condition = ob_t1[0] > 0.5
        
        x_t, y_t = self.get_state_coordinates(ob_t)
        x_t1, y_t1 = self.get_state_coordinates(ob_t1)

        if is_done and terminal_condition:  #Allow for terminal states
           self._tabular_q[x_t][y_t][action] = reward

        else: # Adjust Q value for current state
            self._tabular_q[x_t][y_t][action] += self._alpha*(reward + self._gamma*np.max(self._tabular_q[x_t1][y_t1])-self._tabular_q[x_t][y_t][action])
        
        # P1-5 only
        if is_done and self._epsilon > 0.0:
            # comment these lines to remove epsilon decay (questions 1.1 to 1.4)
            # self._epsilon -= 0.00005
            self._epsilon -= -0.0001

    def get_state_coordinates(self, obs):
        '''
        Returns grid coordinates of state related to obs
        '''

        position = obs[0]
        velocity = obs[1]
        
        x = np.digitize(position, self.position_grid)-1
        y = np.digitize(velocity, self.velocity_grid)-1

        return x, y
