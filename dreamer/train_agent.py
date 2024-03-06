import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import torch
from parse_args import parse_args
from networks import *
from env import Env
from dreamer_agent import Dreamer
from replay_buffer import ReplayBuffer
from utils import lineplot

def initialize_dataset(env, nb_data_collection_episodes):
    '''Initialize replay buffer with S random seed episodes'''
    for s in tqdm(range(1, nb_data_collection_episodes + 1)):
        observation, done, t = env.reset(), False, 0
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done = env.step(action)
            replay_buffer.store_transition(next_observation, action, reward, done)
            observation = next_observation
            t += 1
        metrics['env_steps'].append(t * env._action_repeat + (0 if len(metrics['env_steps']) == 0 else metrics['env_steps'][-1]))
        metrics['episodes'].append(s)
        print("(random)episodes: {}, total_env_steps: {} ".format(metrics['episodes'][-1], metrics['env_steps'][-1]))

def test_models(agent, test_episodes, env, belief_size, state_size, action_size, max_len_episode):
    # Set models to eval mode
    agent.transition_model.eval()
    agent.observation_model.eval()
    agent.reward_model.eval()
    agent.encoder.eval()
    agent.actor_model.eval()
    agent.value_model.eval()
    with torch.no_grad():
        total_reward = 0
        for _ in tqdm(range(test_episodes)):
            obs = env.reset()
            belief = torch.zeros(1, belief_size, device=args.device)
            posterior_state = torch.zeros(1, state_size, device=args.device)
            action = torch.zeros(1, action_size, device=args.device)

            pbar = tqdm(range(max_len_episode // env._action_repeat))
            for t in pbar:
                belief, posterior_state = agent.infer_state(obs, action, belief, posterior_state, device=args.device)
                action = agent.select_action((belief, posterior_state), deterministic=True)

                # interact with environment
                next_obs, reward, done = env.step(action)
                total_reward += reward
                obs = next_obs
                if args.render:
                    env.render()
                if done:
                    pbar.close()
                    break
    print('Average Reward:', total_reward / test_episodes)
    env.close()
    quit()

def train_models(agent, env, metrics, episodes, replay_buffer, collect_interval, belief_size, state_size, action_size, max_len_episode, results_dir, test_interval, checkpoint_interval, checkpoint_experience, test_episodes=1):
    for episode in tqdm(range(metrics['episodes'][-1]+1, episodes + 1), total=episodes, initial=metrics['episodes'][-1]+1):
        data = replay_buffer.get_batches()
        # Model fitting
        loss_info = agent.update_parameters(data, collect_interval)
        # Update and plot loss metrics
        losses = tuple(zip(*loss_info))
        metrics['observation_loss'].append(losses[0])
        metrics['reward_loss'].append(losses[1])
        metrics['kl_loss'].append(losses[2])
        metrics['pcont_loss'].append(losses[3])
        metrics['actor_loss'].append(losses[4])
        metrics['value_loss'].append(losses[5])
        lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss',
                results_dir)
        lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
        lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'kl_loss', results_dir)
        lineplot(metrics['episodes'][-len(metrics['pcont_loss']):], metrics['pcont_loss'], 'pcont_loss', results_dir)
        lineplot(metrics['episodes'][-len(metrics['actor_loss']):], metrics['actor_loss'], 'actor_loss', results_dir)
        lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'value_loss', results_dir)

        # Data collection
        with torch.no_grad():
            observation, total_reward = env.reset(), 0
            belief = torch.zeros(1, belief_size, device=args.device)
            posterior_state = torch.zeros(1, state_size, device=args.device)
            action = torch.zeros(1, action_size, device=args.device)

            pbar = tqdm(range(max_len_episode // env._action_repeat))
            for t in pbar:
                # maintain belief and posterior_state
                belief, posterior_state = agent.infer_state(observation.to(device=args.device), action, belief, posterior_state)
                action = agent.select_action((belief, posterior_state), deterministic=False)

                # interact with env
                next_observation, reward, done = env.step(action)  # Perform environment step (action repeats handled internally)

                # agent.D.append(observation, action.cpu(), reward, done)
                replay_buffer.store_transition(next_observation, action, reward, done)
                total_reward += reward
                observation = next_observation
                
                if args.render:
                    env.render()
                if done:
                    pbar.close()
                    break

            # Update and plot train reward metrics
            metrics['env_steps'].append(t * env._action_repeat + metrics['env_steps'][-1])
            metrics['episodes'].append(episode)
            metrics['train_rewards'].append(total_reward)
            lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards',
                    results_dir)
            print('episode', episode, 'R:', total_reward)

        # Test model
        if episode % test_interval == 0:

            # Set models to eval mode
            agent.transition_model.eval()
            agent.observation_model.eval()
            agent.reward_model.eval()
            agent.encoder.eval()
            agent.actor_model.eval()
            agent.value_model.eval()
            with torch.no_grad():
                observation = env.reset()
                total_rewards = np.zeros((test_episodes, ))

                belief = torch.zeros(test_episodes, belief_size, device=args.device)
                posterior_state = torch.zeros(test_episodes, state_size, device=args.device)
                action = torch.zeros(test_episodes, action_size, device=args.device)

                for t in tqdm(range(max_len_episode // env._action_repeat)):
                    belief, posterior_state = agent.infer_state(observation.to(device=args.device), action, belief, posterior_state)
                    action = agent.select_action((belief, posterior_state), deterministic=True)
                    # interact with env
                    next_observation, reward, done = env.step(action)  # Perform environment step (action repeats handled internally)
                    total_rewards += reward.numpy()
                    observation = next_observation
                    if done.sum().item() == test_episodes:
                        pbar.close()
                        break
            # Update and plot reward metrics (and write video if applicable) and save metrics
            metrics['test_episodes'].append(episode)
            # metrics['test_rewards'].append(total_rewards.tolist())
            metrics['test_rewards'].append(total_rewards)
            lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
            lineplot(np.asarray(metrics['env_steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'],
                    'test_rewards_steps', results_dir, xaxis='env_step')
            torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

            # Set models to train mode
            agent.transition_model.train()
            agent.observation_model.train()
            agent.reward_model.train()
            agent.encoder.train()
            agent.actor_model.train()
            agent.value_model.train()
        
        print("episodes: {}, total_env_steps: {}, train_reward: {} ".format(metrics['episodes'][-1], metrics['env_steps'][-1], metrics['train_rewards'][-1]))
        # Checkpoint models
        if episode % checkpoint_interval == 0:
            torch.save({'transition_model': agent.transition_model.state_dict(),
                        'observation_model': agent.observation_model.state_dict(),
                        'reward_model1': agent.reward_model.state_dict(),
                        'encoder': agent.encoder.state_dict(),
                        'actor_model': agent.actor_model.state_dict(),
                        'value_model1': agent.value_model.state_dict(),
                        'world_optimizer': agent.world_optimizer.state_dict(),
                        'actor_optimizer': agent.actor_optimizer.state_dict(),
                        'value_optimizer': agent.value_optimizer.state_dict()
                        }, os.path.join(results_dir, 'models_%d.pth' % episode))
            if checkpoint_experience:
                torch.save(replay_buffer, os.path.join(results_dir,
                                            'experience_{}.pth'.format(env)))  # Warning: will fail with MemoryError with large memory sizes
    env.close()


# Hyperparameters
args = parse_args()

# Setup
results_dir = os.path.join('results', args.env, str(args.seed))
os.makedirs(results_dir, exist_ok=True)
summary_name = results_dir + "/{}_{}_log"

# set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
else:
  args.device = torch.device('cpu')

metrics = {'env_steps': [], 
           'episodes': [], 
           'train_rewards': [], 
           'test_episodes': [], 
           'test_rewards': [],
           'observation_loss': [], 
           'reward_loss': [], 
           'kl_loss': [], 
           'pcont_loss': [], 
           'actor_loss': [], 
           'value_loss': []}

# Initialise training environment and experience replay memory
env = Env(args.env, args.seed, args.max_episode_length, args.action_repeat)
args.observation_size, args.action_size = env.observation_size, env.action_size

# Initialise agent
agent = Dreamer(args)
replay_buffer = ReplayBuffer(args.observation_size, args.action_size, args.experience_size, args.sequence_length, args.batch_size, args.device)

initialize_dataset(env, args.seed_episodes)
print("--- Finish random data collection  --- ")

if args.test:
    test_models(agent, args.test_episodes, env, args.belief_size, args.state_size, args.action_size, args.max_episode_length)

train_models(agent, env, metrics, args.episodes, replay_buffer, args.collect_interval, args.belief_size, \
             args.state_size, args.action_size, args.max_episode_length, results_dir, args.test_interval, \
             args.checkpoint_interval, args.checkpoint_experience, args.test_episodes)


