import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

exp_path = os.path.abspath('./pendulum con ajustes')
# exp_path = os.path.abspath('./pendulum sin ajustes')
# exp_path = os.path.abspath('./cartpole')

def plot_and_save(dataframe, filename):
    reward = dataframe.loc[:, 'avg'].tolist()
    std = dataframe.loc[:, 'std'].tolist()
    total_steps = dataframe.loc[:, 'steps_1'].tolist()
    plt.errorbar(total_steps, reward, yerr=std, elinewidth=0.5, ecolor='dodgerblue', marker='.',color='C0')
    plt.ylabel('Average Reward')
    plt.xlabel('Steps')
    plt.grid('on')
    plt.savefig(filename)
    plt.pause(0.05)
    plt.close()

if __name__ == '__main__':

    print('\nChoose the environment:\n')
    choice = int(input('To select Pendulum enter 0, to select CartPole enter 1: '))
    if choice == 0: # pendulum
        for i in range(1,4):
            filename = 'pendulum_exp_1' + str(i) + '.pdf'
            df_list = []
            for j in range(1,4):
                path_exp_csv = exp_path + '\pendulum_exp_' + str(i) + '1' + '_' + str(j)
                df = pd.read_csv(path_exp_csv + '.csv', sep='\t')
                df = df.rename(columns={'avg_reward':'avg_reward_' + str(j), 'std_reward':'std_reward_' + str(j), 'steps': 'steps_' + str(j)})
                df_list.append(df)
            full_df = pd.concat(df_list, axis=1)
            full_df['avg'] = full_df[['avg_reward_1', 'avg_reward_2', 'avg_reward_3']].mean(axis=1)
            full_df['std'] = full_df[['avg_reward_1', 'avg_reward_2', 'avg_reward_3']].std(axis=1)
            plot_and_save(full_df, filename)

    elif choice == 1:
        for i in range(1,4):
            filename = 'cartpole_exp_1' + str(i) + '.pdf'
            df_list = []
            for j in range(1,4):
                path_exp_csv = exp_path + '\cartpole_exp_1' + str(i) + '_' + str(j)
                df = pd.read_csv(path_exp_csv + '.csv', sep='\t')
                df = df.rename(columns={'avg_reward':'avg_reward_' + str(j), 'std_reward':'std_reward_' + str(j), 'steps': 'steps_' + str(j)})
                df_list.append(df)
            full_df = pd.concat(df_list, axis=1)
            full_df['avg'] = full_df[['avg_reward_1', 'avg_reward_2', 'avg_reward_3']].mean(axis=1)
            full_df['std'] = full_df[['avg_reward_1', 'avg_reward_2', 'avg_reward_3']].std(axis=1)
            plot_and_save(full_df, filename)