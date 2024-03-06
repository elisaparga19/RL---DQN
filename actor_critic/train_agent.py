import gym
import time
import datetime
import csv
from parse_args import parse_args

import numpy as np
import matplotlib.pyplot as plt

from actor_critic import ActorCriticAgent

def perform_single_rollout(env, agent, render=False):

    # Modify this function to return a tuple of numpy arrays containing:
    # (np.array(obs_t), np.array(acs_t), np.arraw(rew_t), np.array(obs_t1), np.array(done_t))
    # np.array(obs_t)   -> shape: (time_steps, nb_obs)
    # np.array(obs_t1)  -> shape: (time_steps, nb_obs)
    # np.array(acs_t)   -> shape: (time_steps, nb_acs) if actions are continuous, (time_steps,) if actions are discrete
    # np.array(rew_t)   -> shape: (time_steps,)
    # np.array(done_t)  -> shape: (time_steps,)

    ob_t = env.reset()
    
    done = False
    episode_reward = 0
    nb_steps = 0

    obs_t = []
    obs_t1 = []
    acs_t = []
    rew_t = []
    done_t = []
    while not done:

        if render:
            env.render()
            time.sleep(1. / 60)

        obs_t.append(ob_t)
        action = agent.select_action(ob_t)
        acs_t.append(action)                
        ob_t1, reward, done, _ = env.step(action)
        rew_t.append(reward)
        done_t.append(done)
        ob_t1 = np.squeeze(ob_t1)
        obs_t1.append(ob_t1)
        ob_t = ob_t1
        episode_reward += reward
        
        nb_steps += 1

        if done:
            print('Steps: %4d | Reward: %4d' % (nb_steps, episode_reward))
    
    return (np.array(obs_t), np.array(acs_t), np.array(rew_t), np.array(obs_t1), np.array(done_t))


def sample_rollouts(env, agent, training_iter, min_batch_steps):

    sampled_rollouts = []
    total_nb_steps = 0
    episode_nb = 0
    
    while total_nb_steps < min_batch_steps:

        episode_nb += 1
        #render = training_iter%10 == 0 and len(sampled_rollouts) == 0

        # Use perform_single_rollout to get data 
        # Uncomment once perform_single_rollout works.
        # Return sampled_rollouts
        sample_rollout = perform_single_rollout(env, agent)
        total_nb_steps += len(sample_rollout[0])
        sampled_rollouts.append(sample_rollout)
        
    return sampled_rollouts


def train_agent(env, agent, training_iterations, min_batch_steps, nb_critic_updates, filename):

    tr_iters_vec, avg_reward_vec, std_reward_vec, avg_steps_vec = [], [], [], []
    _, (axes) = plt.subplots(1, 2, figsize=(12,4))

    for tr_iter in range(training_iterations + 1):

        # Sample rollouts using sample_rollouts
        sampled_rollouts = sample_rollouts(env, agent, training_iterations, min_batch_steps)

        # performed_batch_steps >= min_batch_steps
        # Parse samples into the following arrays:

        sampled_obs_t  = np.concatenate([r[0] for r in sampled_rollouts], axis=0)   # sampled_obs_t:  Numpy array, shape: (performed_batch_steps, nb_observations)
        sampled_acs_t  = np.concatenate([r[1] for r in sampled_rollouts], axis=0)   # sampled_acs:    Numpy array, shape: (performed_batch_steps, nb_actions) if actions are continuous, 
                                                                                    #(performed_batch_steps,) if actions are discrete
        sampled_rew_t  = np.concatenate([r[2] for r in sampled_rollouts], axis=0)   # sampled_rew_t:  Numpy array, shape: (performed_batch_steps,)
        sampled_obs_t1 = np.concatenate([r[3] for r in sampled_rollouts], axis=0)   # sampled_obs_t1: Numpy array, shape: (performed_batch_steps, nb_observations)
        sampled_done_t = np.concatenate([r[4] for r in sampled_rollouts], axis=0)   # sampled_done_t: Numpy array, shape: (performed_batch_steps,)

        # performance metrics
        update_performance_metrics(tr_iter, sampled_rollouts, axes, tr_iters_vec, avg_reward_vec, std_reward_vec, avg_steps_vec)

        for _ in range(nb_critic_updates):
            agent.update_critic(sampled_obs_t, sampled_rew_t, sampled_obs_t1, sampled_done_t)
        
        agent.update_actor(sampled_obs_t, sampled_acs_t, sampled_rew_t, sampled_obs_t1, sampled_done_t)

    save_metrics(tr_iters_vec, avg_reward_vec, std_reward_vec, filename)


def update_performance_metrics(tr_iter, sampled_rollouts, axes, tr_iters_vec, avg_reward_vec, std_reward_vec, avg_steps_vec):

    raw_returns     = np.array([np.sum(rollout[2]) for rollout in sampled_rollouts])
    rollout_steps   = np.array([len(rollout[2]) for rollout in sampled_rollouts])

    avg_return = np.average(raw_returns)
    max_episode_return = np.max(raw_returns)
    min_episode_return = np.min(raw_returns)
    std_return = np.std(raw_returns)
    avg_steps = np.average(rollout_steps)

    # logs 
    print('-' * 32)
    print('%20s : %5d'   % ('Training iter'     ,(tr_iter)              ))
    print('-' * 32)
    print('%20s : %5.3g' % ('Max episode return', max_episode_return    ))
    print('%20s : %5.3g' % ('Min episode return', min_episode_return    ))
    print('%20s : %5.3g' % ('Return avg'        , avg_return            ))
    print('%20s : %5.3g' % ('Return std'        , std_return            ))
    print('%20s : %5.3g' % ('Steps avg'         , avg_steps             ))

    avg_reward_vec.append(avg_return)
    std_reward_vec.append(std_return)

    avg_steps_vec.append(avg_steps)

    tr_iters_vec.append(tr_iter)

    plot_performance_metrics(axes, 
                            tr_iters_vec, 
                            avg_reward_vec, 
                            std_reward_vec, 
                            avg_steps_vec)


def plot_performance_metrics(axes, tr_iters_vec, avg_reward_vec, std_reward_vec, avg_steps_vec):
    ax1, ax2 = axes
    
    [ax.cla() for ax in axes]
    ax1.errorbar(tr_iters_vec, avg_reward_vec, yerr=std_reward_vec, marker='.',color='C0')
    ax1.set_ylabel('Avg Reward')
    ax2.plot(tr_iters_vec, avg_steps_vec, marker='.',color='C1')
    ax2.set_ylabel('Avg Steps')

    [ax.grid('on') for ax in axes]
    [ax.set_xlabel('training iteration') for ax in axes]
    plt.pause(0.05)


def save_metrics(tr_iters_vec, avg_reward_vec, std_reward_vec, filename):
    with open(filename + '.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerow(['steps', 'avg_reward', 'std_reward'])
        for i in range(len(tr_iters_vec)):
            csv_writer.writerow([tr_iters_vec[i], avg_reward_vec[i], std_reward_vec[i]])


if __name__ == '__main__':

    #env = gym.make('Pendulum-v1')
    #env = gym.make('CartPole-v1')

    args = parse_args()

    env_name = args.env
    training_iterations = args.training_iterations
    batch_size = args.batch_size
    nb_critic_updates = args.nb_critic_updates
    critic_lr = args.critic_lr
    filename = args.filename

    envs = {'CartPole-v1': gym.make('CartPole-v1'), 'Pendulum-v1': gym.make('Pendulum-v1'), 'Acrobot-v1': gym.make('Acrobot-v1')}
    env = envs[env_name]

    dim_states = env.observation_space.shape[0]

    continuous_control = isinstance(env.action_space, gym.spaces.Box)

    dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n

    actor_critic_agent = ActorCriticAgent(dim_states=dim_states,
                                          dim_actions=dim_actions,
                                          actor_lr=0.001,
                                          critic_lr=critic_lr,
                                          gamma=0.99,
                                          continuous_control=continuous_control)

    train_agent(env=env, 
                agent=actor_critic_agent, 
                training_iterations=training_iterations, 
                min_batch_steps=batch_size,
                nb_critic_updates=nb_critic_updates,
                filename=filename)
