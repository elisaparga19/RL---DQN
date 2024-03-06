from train_agent import *
from actor_critic import ActorCriticAgent

def select_test():
    print('\nChoose the environment:\n')
    choice = int(input('To select Pendulum enter 1, to select CartPole enter 0: '))
    if choice == 1:
        return gym.make('Pendulum-v1')
    else:
        return gym.make('CartPole-v1')


if __name__ == '__main__':

    env = select_test()
    dim_states = env.observation_space.shape[0]

    continuous_control = isinstance(env.action_space, gym.spaces.Box)
    dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n

    actor_critic_agent = ActorCriticAgent(dim_states=dim_states, 
                                             dim_actions=dim_actions, 
                                             actor_lr=0.005,
                                             critic_lr=0.005,
                                             gamma=0.99,
                                             continuous_control=continuous_control)
    
    rollout = perform_single_rollout(env, actor_critic_agent)
    rollout_steps = len(rollout[0])
    assert rollout[0].shape == (rollout_steps, dim_states), 'Shape of obs_t incorrect'
    assert rollout[3].shape == (rollout_steps, dim_states), 'Shape of obs_t1 incorrect'
    assert rollout[1].shape == (rollout_steps, dim_actions) if continuous_control else (rollout_steps,), 'Shape of actions incorrect'
    assert rollout[2].shape == (rollout_steps,), 'Shape of rewards incorrect'
    assert rollout[4].shape == (rollout_steps,), 'Shape of done_t incorrect'
    print('\nType of perform_single_rollout(): ', type(rollout))
    print('Length rollout: ', rollout_steps)
    print('Shape obs_t: ', rollout[0].shape)
    print('Shape obs_t1: ', rollout[3].shape)
    print('Shape actions: ', rollout[1].shape)
    print('Shape rewards: ', rollout[2].shape)
    print('Shape done: ', rollout[4].shape)

    print('\nNow we are going to test the sample_rollouts() function: \n')
    input('Press enter to continue \n')
    min_batch_size = 5000
    sampled_rollouts = sample_rollouts(env, actor_critic_agent, 1000, min_batch_size)
    # test_1 = np.concatenate([r[0] for r in sampled_rollouts], axis=0)
    # print(test_1.shape)
    # test_2 = np.concatenate([r[1] for r in sampled_rollouts], axis=0)
    # print(test_2.shape)
    # test_3 = np.concatenate([r[2] for r in sampled_rollouts], axis=0)
    # print(test_3.shape)
    # test_4 = np.concatenate([r[4] for r in sampled_rollouts], axis=0)
    # print(test_4.shape)
    assert sum([len(sample[0]) for sample in sampled_rollouts]) >= min_batch_size, 'Not enough samples were sampled'
    print(len(sampled_rollouts), ' rollouts were sampled')
    
