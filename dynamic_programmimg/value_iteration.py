import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from grid_world import GridWorld
from utils import display_policy
from utils import display_value_function


class ValueIterator():

    def __init__(self, reward_grid, wall_value, cell_value, terminal_value):

        self._reward_grid = reward_grid
        self._wall_value = wall_value
        self._cell_value = cell_value
        self._terminal_value = terminal_value

        self._value_function = np.zeros(self._reward_grid.shape)
        self._value_function *= self._reward_grid
        self._policy = self._value_function.copy()


    def run_value_iteration(self, p_dir, nb_iters, gamma, v_thresh):
        p_random    = 1 - p_dir
        p_side = p_random/2
        value_rows, value_cols = self._value_function.shape
        # Notice that in the reward grid walls are nans, traversable cells are -1's and the goal is 0.
        
        # V(s) = max_a sum_s'(P_ss'^a[R_ss'^a+gamma*V(s')]) 
        # We perform all posible actions, check the value, and update according to the max value found
        counter = 0
        for _ in tqdm(range(nb_iters)):
            counter += 1
            delta_v = 0
            # Indexes for skipping external walls (you may change them)
            for j in range(value_rows):
                for i in range(value_cols):

                    if np.isnan(self._reward_grid[j][i]): # verify if the cell corresponds to a wall
                        continue
                    
                    r = self._reward_grid[j][i] # reward at current state
                    val = self._value_function[j][i] # value function at current state
                    
                    if r == 0: # agent in the terminal state
                        value_function = 0
                    else:
                        possible_values = np.zeros(4)
                        for possible_action in range(4):
                            value_function = self.compute_value_function(gamma, p_dir, p_side, j, i, r, val, possible_action)
                            possible_values[possible_action] = value_function
                        
                        new_action = np.argmax(possible_values)
                        self._policy[j][i] = new_action

                        value_function = max(possible_values)
                        self._value_function[j][i] = value_function                 

                    v_diff = abs(val-value_function)
                    delta_v = max(delta_v, v_diff)
            
            if delta_v < v_thresh: # with this condition we take the optimal value function
                break
        print(counter)

    def compute_value_function(self, gamma, p_dir, p_side, j, i, r, val, action):
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
        # array with perpendicular directions to action (0: up, 1: down, 2: right, 3: left)
        perpendicular_actions = self.get_perpendicular_direction(action) 

        # arrays with value function and reward for the states that can be reached with all possible actions from state (j, i), sorted by actions (up, down, right, left)
        val_next_state = [self._value_function[j-1][i], self._value_function[j+1][i], self._value_function[j][i+1], self._value_function[j][i-1]]
        reward_next_state = [self._reward_grid[j-1][i], self._reward_grid[j+1][i], self._reward_grid[j][i+1], self._reward_grid[j][i-1]]

        exp_val = 0 # expectation term of value function
        exp_val += p_dir*(val if np.isnan(reward_next_state[action]) else val_next_state[action])
        exp_val += p_side*(val if np.isnan(reward_next_state[perpendicular_actions[0]]) else val_next_state[perpendicular_actions[0]])
        exp_val += p_side*(val if np.isnan(reward_next_state[perpendicular_actions[1]]) else val_next_state[perpendicular_actions[1]])

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

    value_iterator = ValueIterator(reward_grid=world._rewards,
                                   wall_value=None,
                                   cell_value=-1,
                                   terminal_value=0)

    # Default parameters for P2-2 (change them for P2-3 & P2-4 & P2-5)
    value_iterator.run_value_iteration(p_dir=0.6,
                                       nb_iters=1000,
                                       gamma=1,
                                       v_thresh=0.0001)

    world.display()

    display_value_function(value_iterator._value_function)

    display_policy(world._grid,
                   value_iterator._reward_grid,
                   value_iterator._policy)

    plt.show()
