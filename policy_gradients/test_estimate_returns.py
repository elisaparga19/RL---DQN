from train_agent import *
from test_samples import select_test

env_1 = gym.make('Pendulum-v1')
env_2 = gym.make('CartPole-v1')

def test_estimate_returns(env, n_samples):

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

    rollouts_rew = []
    if n_samples == 1:
        rollouts_rew.append(perform_single_rollout(env, policy_gradients_agent, 1)[2])
    elif n_samples == 2:
        rollouts_rew.append(perform_single_rollout(env, policy_gradients_agent, 1)[2])
        rollouts_rew.append(perform_single_rollout(env, policy_gradients_agent, 1)[2])
    else:
        print('Enter 1 or 2 as n_samples')

    est_rw_1 = policy_gradients_agent.estimate_returns(rollouts_rew)
    total_steps = range(len(est_rw_1))
    return total_steps, est_rw_1


if __name__ == '__main__':

    env = select_test()
    print('\nChoose a number of rolloutss:\n')
    n_samples = int(input('To select 1 enter 1, to select 2 enter 2: '))
    total_steps, est_rw_1 = test_estimate_returns(env, n_samples)

    plt.plot(total_steps, est_rw_1, marker='.',color='C0')
    plt.ylabel('Estimate Returns')
    plt.xlabel('Steps')
    plt.grid('on')
    plt.pause(0.5)
    #plt.savefig('enstimate_returns_pendulum_' + str(n_samples) + '.pdf')
    plt.close()