import numpy as np

class Buffer:

    def __init__(self, dim_states, dim_actions, max_size, sample_size):

        assert sample_size < max_size, "Sample size cannot be greater than buffer size"
        
        self._buffer_idx     = 0
        self._exps_stored    = 0
        self._buffer_size    = max_size
        self._sample_size    = sample_size

        self._s_t_array      = np.zeros(shape=(self._buffer_size, dim_states))
        self._a_t_array      = np.zeros(shape=(self._buffer_size))
        self._s_t1_array     = np.zeros(shape=(self._buffer_size, dim_states))


    def store_transition(self, s_t, a_t, s_t1):
        # Add transition to the buffer
        self._s_t_array[self._buffer_idx] = s_t
        self._s_t1_array[self._buffer_idx] = s_t1
        if isinstance(a_t, list):
            self._a_t_array[self._buffer_idx] = a_t[0]
        else:
            self._a_t_array[self._buffer_idx] = a_t

        # Update replay buffer index
        self._buffer_idx = (self._buffer_idx + 1) % self._buffer_size
        self._exps_stored = min(self._exps_stored + 1, self._buffer_size)

    
    def get_batches(self):
        assert self._exps_stored + 1 > self._sample_size, "Not enough samples has been stored to start sampling"
        # Get all the data contained in the buffer as batches
        batches = []
        for i in range(0, self._exps_stored, self._sample_size):
            batch_s_t = self._s_t_array[i : i+self._sample_size]
            batch_a_t = self._a_t_array[i : i+self._sample_size]
            batch_s_t1 = self._s_t1_array[i : i+self._sample_size]
            batch = (batch_s_t, batch_a_t, batch_s_t1)
            batches.append(batch)
        return batches