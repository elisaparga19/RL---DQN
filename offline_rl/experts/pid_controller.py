import gym
import time
import numpy as np

class PIDController:

    def __init__(self, kp, ki, kd, dt=0.02):
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._dt = dt
        
        # P1-1
        # Define aux variables (if any)
        self._error = 0
        self._integral = 0
        
    def select_action(self, observation):
        # P1-1
        # Set point (do not change)
        error = observation[2]
        proportional = error
        self._integral += error*self._dt
        derivative = (error - self._error)/self._dt

        # PID control
        # Code the PID control law
        ctrl = (self._kp * proportional) + (self._ki * self._integral) + (self._kd * derivative)
        self._error = error
        return 0 if ctrl < 0 else 1


def test_agent(env, agent, nb_episodes=30, render=False):

    ep_rewards = []
    success_rate = 0
    avg_steps = 0

    for episode in range(nb_episodes):

        ob_t = env.reset()
        done = False
        episode_reward = 0
        nb_steps = 0

        while not done:

            if render and episode == 0:
                env.render()
                time.sleep(1. / 60)
                
            action = agent.select_action(ob_t)
            
            ob_t1, reward, done, _ = env.step(action)

            ob_t = ob_t1
            episode_reward += reward
            
            nb_steps += 1

            if done:
                if nb_steps == 200:
                    success_rate += 1.
                avg_steps += nb_steps
                ep_rewards.append(episode_reward)
                print('Evaluation episode %3d | Steps: %4d | Reward: %4d | Success: %r' % (episode + 1, nb_steps, episode_reward, nb_steps == 200))
    
    ep_rewards = np.array(ep_rewards)
    avg_reward = np.average(ep_rewards)
    std_reward = np.std(ep_rewards)
    success_rate /= nb_episodes
    avg_steps /= nb_episodes
    print('Average Reward: %.2f| Reward Deviation: %.2f | Average Steps: %.2f| Success Rate: %.2f' % (avg_reward, std_reward, avg_steps, success_rate))
    return avg_reward


def grid_search(env, agent, param_ranges, nb_episodes=30, render=False):

    best_reward = -np.inf
    best_params = None

    # Generate all parameter combinations
    param_combinations = np.array(np.meshgrid(*param_ranges)).T.reshape(-1, len(param_ranges))

    # Iterate over parameter combinations
    for params in param_combinations:
        agent = PIDController(*params)  # Set PID parameters
        
        # Test the agent with the current parameters
        avg_reward = test_agent(env, agent, nb_episodes, render=render)
        
        # Update the best parameters if necessary
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = params

    return best_params, best_reward


# P1-2, P1-3
if __name__ == '__main__':

    # do not change dt = 0.02
    env = gym.make('CartPole-v0')
    param_ranges = [
    np.linspace(0.0, 2.0, num=10),  # Range for kp
    np.linspace(0.0, 0.2, num=10),  # Range for ki
    np.linspace(0.0, 0.2, num=10)  # Range for kd
    ]
    pid_agent = PIDController(0.1, 0.01, 0.01, 0.02)
    #best_params, best_reward = grid_search(env, pid_agent, param_ranges, nb_episodes=30, render=False)
    #print("Best Parameters:", best_params)
    #print("Best Reward:", best_reward)
    test_agent(env, pid_agent, render=False)
