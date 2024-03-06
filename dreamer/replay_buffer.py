import numpy as np
import torch

class ReplayBuffer:

    def __init__(self, dim_obs, dim_actions, max_size, sequence_length, batch_size, device):

        assert sequence_length < max_size, "Sample size cannot be greater than buffer size"
        self.device = device
        self._buffer_size = max_size
        self._sequence_length = sequence_length
        self._batch_size = batch_size
        self._buffer_idx = 0
        self._full = False # Tracks if memory has been filled/all slots are valid

        self.observations      = np.empty(shape=(self._buffer_size, dim_obs), dtype=np.float32)
        self.actions      = np.empty(shape=(self._buffer_size, dim_actions), dtype=np.float32)
        self.rewards      = np.empty(shape=(self._buffer_size, ), dtype=np.float32)
        self.nonterminals   = np.empty(shape=(self._buffer_size, 1), dtype=np.float32)

        self._steps, self._episodes = 0, 0 # Tracks how much experience has been used in total


    def store_transition(self, observation, action, reward, done):

        # Add transition to replay buffer
        self.observations[self._buffer_idx]     = observation
        self.actions[self._buffer_idx]     = action.numpy()
        self.rewards[self._buffer_idx]     = reward
        self.nonterminals[self._buffer_idx]  = not done

        # Update replay buffer index
        self._buffer_idx = (self._buffer_idx + 1) % self._buffer_size
        self._full = self._full or self._buffer_idx == 0

        self._steps += 1
        self._episodes += (1 if done else 0)

    def get_sequence_idxs(self):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(low = 0, high = self._buffer_size if self._full else self._buffer_idx-self._sequence_length)
            sequence_idxs = np.arange(idx, idx + self._sequence_length) % self._buffer_size
            valid_idx = not self._buffer_idx in sequence_idxs[1:]
        return sequence_idxs

    def get_batches(self):
        assert self._steps + 1 > self._sequence_length, "Not enough samples has been stored to start sampling"

        batch_idx = np.asarray([self.get_sequence_idxs() for _ in range(self._batch_size)]) # generate a matrix of size (batch_size, sequence_length) with index
        batch_idx = batch_idx.transpose().reshape(-1)
        observations = self.observations[batch_idx]

        batches = (observations.reshape(self._sequence_length, self._batch_size, *observations.shape[1:]),
                self.actions[batch_idx].reshape(self._sequence_length, self._batch_size, -1),
                self.rewards[batch_idx].reshape(self._sequence_length, self._batch_size),
                self.nonterminals[batch_idx].reshape(self._sequence_length, self._batch_size, 1))

        return [torch.as_tensor(item).to(device=self.device) for item in batches]
    
