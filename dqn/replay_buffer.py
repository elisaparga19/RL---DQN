import numpy as np
import os

class ReplayBuffer:

    def __init__(self, dim_states, dim_actions, max_size, sample_size):

        assert sample_size < max_size, "Sample size cannot be greater than buffer size"
        
        self._buffer_idx     = 0
        self._exps_stored    = 0
        self._buffer_size    = max_size
        self._sample_size    = sample_size
        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._s_t_array      = np.zeros(shape=(self._buffer_size, self._dim_states))
        self._a_t_array      = np.zeros(shape=self._buffer_size)
        self._r_t_array      = np.zeros(shape=self._buffer_size)
        self._s_t1_array     = np.zeros(shape=(self._buffer_size, self._dim_states))
        self._term_t_array   = np.zeros(shape=self._buffer_size)


    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):

        # Add transition to replay buffer according to self._buffer_idx
        self._s_t_array[self._buffer_idx] = s_t
        self._a_t_array[self._buffer_idx] = a_t
        self._r_t_array[self._buffer_idx] = r_t
        self._s_t1_array[self._buffer_idx] = s_t1
        self._term_t_array[self._buffer_idx] = done_t

        # Update replay buffer index
        self._buffer_idx = (self._buffer_idx + 1) % self._buffer_size
        self._exps_stored = min(self._exps_stored + 1, self._buffer_size)
    

    def sample_transitions(self):
        assert self._exps_stored + 1 > self._sample_size, "Not enough samples have been stored to start sampling"
        
        sample_idxs = np.random.randint(0, self._exps_stored, self._sample_size)
        # np.random.choice
        
        return (self._s_t_array[sample_idxs],
                self._a_t_array[sample_idxs],
                self._r_t_array[sample_idxs],
                self._s_t1_array[sample_idxs],
                self._term_t_array[sample_idxs])


# Test replay buffer
if __name__ == '__main__':

    test_RB = ReplayBuffer(2, 3, 5, 3)
    print('\n' + '-'*100 + '\n')
    print('For the following test, press enter keyboard to continue when the program is paused')
    input()
    print('\n' + '-'*100 + '\n')
    print('We have created a Replay Buffer with the following parameters: \n')
    print('dim_states = 0')
    print('dim_actions = 3')
    print('max_size = 5')
    print('sample_size = 3 \n')
    input()
    print('\n' + '-'*100 + '\n')
    print('We are going to store 4 experiences \n')
    input()
    # Store 5 transitions
    test_RB.store_transition(np.array([17.0, 12.5]), 2, 100, np.array([15.0, 13.0]), False)
    test_RB.store_transition(np.array([15.0, 13.0]), 2, -100, np.array([15.0, 19.0]), False)
    test_RB.store_transition(np.array([9.0, 1]), 1, 1, np.array([15.0, 19.0]), False)
    test_RB.store_transition(np.array([77.0, 9]), 2, 30, np.array([15.0, 19.0]), False)
    print('There are ' + str(test_RB._exps_stored) + ' experiences stored so far and the buffer index is ' + str(test_RB._buffer_idx))
    input()
    print('\nThe data stored in the buffer: \n')
    print('States: ', test_RB._s_t_array)
    print('Actions: ', test_RB._a_t_array)
    print('Rewards: ', test_RB._r_t_array)
    print('Next States: ', test_RB._s_t1_array)
    print('Terminal State: ', test_RB._term_t_array)
    print('\n' + '-'*100 + '\n')
    input()
    print('We are going to store a new experience: \n')
    input()
    test_RB.store_transition(np.array([8.0, 1.5]), 1, 10, np.array([15.0, 19.0]), False)
    print('\nNow the number of experiences stored and buffer index are updated to ' + str(test_RB._exps_stored) + ' and ' + str(test_RB._buffer_idx) + ' respectively.')
    print('\n' + '-'*100 + '\n')
    input()

    print('Now if we want to add a new experience to the buffer: \n')
    input()
    test_RB.store_transition(np.array([9.0, 12.5]), 0, 200, np.array([8.0, 1.5]), False)
    print('\nThe data stored in the gets updated to: \n')
    print('States: ', test_RB._s_t_array)
    print('Actions: ', test_RB._a_t_array)
    print('Rewards: ', test_RB._r_t_array)
    print('Next States: ', test_RB._s_t1_array)
    print('Terminal State: ', test_RB._term_t_array)
    print('\nThe number of experiences stored is ' + str(test_RB._exps_stored) + ' and the buffer index ' + str(test_RB._buffer_idx))
    print('\n' + '-'*100 + '\n')
    input()
    
    print('Now we are going to test the sample method: \n')
    input()
    sample_1 = test_RB.sample_transitions()
    sample_2 = test_RB.sample_transitions()
    sample_3 = test_RB.sample_transitions()
    
    print("3 examples of samples:\n")
    print(sample_1)
    print(sample_2)
    print(sample_3)
    print('\n' + '-'*100 + '\n')

