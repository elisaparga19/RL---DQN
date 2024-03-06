import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cartpole_path = os.path.abspath('./offline_rl/cartpole')
mountaincar_path = os.path.abspath('./offline_rl/mountaincar')

def plot_and_save(dataframe, filename):
    _, (axes) = plt.subplots(1, 2, figsize=(12,4))
    ax1, ax2 = axes

    total_steps = dataframe.loc[:, 'steps_0'].tolist()
    avg_reward = dataframe.loc[:, 'avg'].tolist()
    std_rw = dataframe.loc[:, 'std_avg'].tolist()
    sr = dataframe.loc[:, 'sr'].tolist()
    
    [ax.cla() for ax in axes]
    ax1.errorbar(total_steps, avg_reward, yerr=std_rw, marker='.', color='C0')
    ax1.set_ylabel('Avg. Return')
    ax2.plot(total_steps, sr, marker='.', color='C1')
    ax2.set_ylabel('Success Rate')

    [ax.grid('on') for ax in axes]
    [ax.set_xlabel('training steps') for ax in axes]

    plt.savefig(filename)
    plt.pause(0.05)
    plt.close()

if __name__ == '__main__':

    print('\nChoose the environment:\n')
    choice = int(input('To select CartPole enter 0, to select MountainCar enter 1: '))
    if choice == 0: # cartpole
        for i in range(1, 4):
            for j in range(1, 5):
                filename = 'cartpole_exp_' + str(i) + str(j)
                path = os.path.join(cartpole_path, filename)
                filename_list = os.listdir(path)
                df_list = []
                for k in range(3):
                    df = pd.read_csv(os.path.join(path, filename_list[k]), sep='\t')
                    df = df.rename(columns={'avg_reward':'avg_reward_' + str(k), 
                                            'std_reward':'std_reward_' + str(k), 
                                            'steps': 'steps_' + str(k), 
                                            'success_rate': 'success_rate_' + str(k)})
                    df_list.append(df)
                full_df = pd.concat(df_list, axis=1)
                full_df['avg'] = full_df[['avg_reward_1', 'avg_reward_2', 'avg_reward_0']].mean(axis=1)
                full_df['std_avg'] = full_df[['avg_reward_1', 'avg_reward_2', 'avg_reward_0']].std(axis=1)
                full_df['sr'] = full_df[['success_rate_1', 'success_rate_2', 'success_rate_0']].mean(axis=1)
                plot_and_save(full_df, filename + '.pdf')

    elif choice == 1: # mountaincar
        for i in range(1, 4):
            for j in range(1, 5):
                filename = 'mountaincar_exp_' + str(i) + str(j)
                path = os.path.join(mountaincar_path, filename)
                filename_list = os.listdir(path)
                df_list = []
                for k in range(3):
                    df = pd.read_csv(os.path.join(path, filename_list[k]), sep='\t')
                    df = df.rename(columns={'avg_reward':'avg_reward_' + str(k), 
                                            'std_reward':'std_reward_' + str(k), 
                                            'steps': 'steps_' + str(k), 
                                            'success_rate': 'success_rate_' + str(k)})
                    df_list.append(df)
                full_df = pd.concat(df_list, axis=1)
                full_df['avg'] = full_df[['avg_reward_1', 'avg_reward_2', 'avg_reward_0']].mean(axis=1)
                full_df['std_avg'] = full_df[['avg_reward_1', 'avg_reward_2', 'avg_reward_0']].std(axis=1)
                full_df['sr'] = full_df[['success_rate_1', 'success_rate_2', 'success_rate_0']].mean(axis=1)
                plot_and_save(full_df, filename + '.pdf')