import numpy as np
import torch
from dm_control import suite
import cv2

GYM_ENVS = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']
CONTROL_SUITE_ENVS = ['cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin', 'cheetah-run', 'ball_in_cup-catch', 'walker-walk','reacher-hard', 'walker-run', 'humanoid-stand', 'humanoid-walk', 'fish-swim', 'acrobot-swingup']
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2, 'humanoid': 2, 'fish': 2, 'acrobot':4}

class Env():
    def __init__(self, env, seed, max_episode_length, action_repeat):
        domain, task = env.split('-')
        self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
        self._max_episode_length = max_episode_length
        self._action_repeat = action_repeat

    def concatenate_observations(self, observations):
        '''Concatenates all the values of the observations'''
        obs_values = observations.values()
        return np.concatenate([np.asarray([obs]) 
                               if isinstance(obs, float) 
                               else obs for obs in obs_values], 
                               axis=0)

    def reset(self):
        self._t = 0 # Reset internal timer
        state = self._env.reset()
        observations = self.concatenate_observations(state.observation)

        # with .unsqueeze() add a dim in 0-axis tensor([1, 2]) --> tensor([[1,2]])
        obs_tensor = torch.tensor(observations, dtype=torch.float32).unsqueeze(dim=0)
        return obs_tensor
    
    def step(self, action):
        action = action.detach().numpy() # transform action tensor to numpy
        reward = 0
        for _ in range(self._action_repeat):
            state = self._env.step(action)
            reward += state.reward
            self._t += 1  # Increment internal timer
            done = state.last() or self._t == self._max_episode_length
            if done:
                break
        observation = torch.tensor(self.concatenate_observations(state.observation), 
                                   dtype=torch.float32).unsqueeze(dim=0)
        return observation, reward, done

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        spec = self._env.action_spec()
        action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
        return torch.from_numpy(action)
    
    def render(self):
        cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
        cv2.waitKey(1)
    
    def close(self):
        cv2.destroyAllWindows()
        self._env.close()
    
    @property
    def observation_size(self):
        return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in self._env.observation_spec().values()])

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]