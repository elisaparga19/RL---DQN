import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from grid_world import GridWorld
from utils import display_policy
from utils import display_value_function

class PolicyIterator():

    def __init__(self, reward_grid, wall_value, cell_value, terminal_value):

        self._reward_grid = reward_grid
        self._wall_value = wall_value
        self._cell_value = cell_value
        self._terminal_value = terminal_value

        self._value_function = np.zeros(self._reward_grid.shape) # zero initialization for value
        self._value_function *= self._reward_grid
        self._policy = self._value_function.copy()

        # To test random initialization
        # self._policy = np.random.randint(4, size=self._reward_grid.shape)


    def _policy_evaluation(self, nb_iters, p_dir, gamma, v_thresh):
        # Policy evaluation        
        p_random    = 1 - p_dir
        value_rows, value_cols = self._value_function.shape
        counter = 0
        for _ in tqdm(range(nb_iters)):              
            counter += 1
            delta_v = 0

            for j in range(value_rows): # iteration over rows
                for i in range(value_cols): # iteration over columns
                    
                    if np.isnan(self._reward_grid[j][i]): # verify if the cell corresponds to a wall
                        continue
                    
                    val = self._value_function[j][i] # value function at current state
                    action = int(self._policy[j][i]) # action of current policy at current state
                    r = self._reward_grid[j][i] # get the reward at current state

                    if r == 0: # case state is the terminal state, we don't update value function
                        value_function = r
                    else:
                        value_function = self.compute_value_function(gamma, p_dir, p_random, j, i, r, val, action)
                        self._value_function[j][i] = value_function # update value function at current state
                    
                    v_diff = np.abs(val - value_function)
                    delta_v = max(delta_v, v_diff) # update delta_v

            if delta_v < v_thresh:
                break
        print(counter)
                        
                    
    def _policy_improvement(self, p_dir, gamma):
        # Policy improvement
        p_random    = 1 - p_dir
        value_rows, value_cols = self._value_function.shape

        stable_policy = True
            
        old_policy = self._policy.copy()

        for j in range(value_rows):
            for i in range(value_cols):

                if np.isnan(self._reward_grid[j][i]): # verify if the cell corresponds to a wall
                        continue

                action = int(old_policy[j][i])
                r = self._reward_grid[j][i] # get the direct reward
                val = self._value_function[j][i] # value function at current state

                if r == 0:
                    continue
                else:
                    possible_value = np.zeros(4) # array to save value functions for all possible actions from current state
                    for possible_action in range(4):
                        
                        value_function = self.compute_value_function(gamma, p_dir, p_random, j, i, r, val, possible_action)
                        possible_value[possible_action] = value_function
                    
                    new_action = np.argmax(possible_value)
                    self._policy[j][i] = new_action
                    
                    if action != new_action:
                        stable_policy = False

        return stable_policy


    def run_policy_iteration(self, p_dir, nb_iters, gamma, v_thresh):
        stable_policy = False

        while not stable_policy:
            self._policy_evaluation(nb_iters, p_dir, gamma, v_thresh)
            stable_policy = self._policy_improvement(p_dir, gamma)

    def compute_value_function(self, gamma, p_dir, p_random, j, i, r, val, action):
        '''
        Computes the value function at state (j, i)
        Args:
            - gamma: discount factor
            - p_dir: probability of following action given by the current policy
            - p_random: 1-p_dir
            - (j, i): indices of state W[j][i]
            - r: reward at state (j, i)
            - val: value function at state (j, i)
            - action: action given by the current policy in to the state (j, i)
        '''
        perpendicular_actions = self.get_perpendicular_direction(action) # array with perpendicular directions to action

        # arrays with value function and reward for the states that can be reached with all possible actions from state (j, i), sorted by actions (up, down, right, left)
        val_next_state = [self._value_function[j-1][i], self._value_function[j+1][i], self._value_function[j][i+1], self._value_function[j][i-1]]
        reward_next_state = [self._reward_grid[j-1][i], self._reward_grid[j+1][i], self._reward_grid[j][i+1], self._reward_grid[j][i-1]]

        exp_val = 0 # expectation term of value function
        exp_val += p_dir*(val if np.isnan(reward_next_state[action]) else val_next_state[action])
        exp_val += (p_random/2)*(val if np.isnan(reward_next_state[perpendicular_actions[0]]) else val_next_state[perpendicular_actions[0]])
        exp_val += (p_random/2)*(val if np.isnan(reward_next_state[perpendicular_actions[1]]) else val_next_state[perpendicular_actions[1]])

        value_function = r + gamma*exp_val
        
        return value_function


    def get_perpendicular_direction(self, action):
        '''
        Returns array with perpendicular directions
        Args:
            - action: action from which perpendicular directions are obtained
        '''
        return [2, 3] if action in [0, 1] else [0, 1]        

if __name__ == '__main__':

    world = GridWorld(height=14, width=16)
    policy_iterator = PolicyIterator(reward_grid=world._rewards,
                                     wall_value=None,
                                     cell_value=-1,
                                     terminal_value=0)

    # Default parameters for P1-3 (change them for P2-3)
    policy_iterator.run_policy_iteration(p_dir=0.8,
                                         nb_iters=1000,
                                         gamma=0.9,
                                         v_thresh=0.0001)

    world.display()

    display_value_function(policy_iterator._value_function)

    display_policy(world._grid,
                   policy_iterator._reward_grid,
                   policy_iterator._policy)

    plt.show()
