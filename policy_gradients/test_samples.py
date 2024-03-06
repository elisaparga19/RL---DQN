from train_agent import *

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

    policy_gradients_agent = PolicyGradients(dim_states=dim_states, 
                                             dim_actions=dim_actions, 
                                             lr=0.005,
                                             gamma=0.99,
                                             continuous_control=continuous_control,
                                             reward_to_go=False,
                                             use_baseline=False)
    
    rollout = perform_single_rollout(env, policy_gradients_agent, 1)
    print('\nType of perform_single_rollout(): ', type(rollout))
    print('Length rollout: ', len(rollout))
    print('Shape observations: ', rollout[0].shape)
    print('Shape actions: ', rollout[1].shape)
    print('Shape rewards: ', rollout[2].shape)

    print('\nNow we are going to test the sample_rollouts() function: \n')
    input('Press enter to continue \n')
    sampled_rollouts = sample_rollouts(env, policy_gradients_agent, 1000, 5000)
    print(len(sampled_rollouts), ' rollouts were sampled')
    
