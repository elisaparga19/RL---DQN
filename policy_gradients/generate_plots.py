import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

exp_path = os.path.abspath('../Tarea 3/policy_gradients')

def plot_and_save(dataframe, filename):
    reward = dataframe.loc[:, 'avg'].tolist()
    total_steps = range(len(dataframe))
    plt.plot(total_steps, reward, marker='.',color='C0')
    plt.ylabel('Average Reward')
    plt.xlabel('Steps')
    plt.grid('on')
    plt.pause(0.05)
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':

    print('\nChoose the environment:\n')
    choice = int(input('To select Pendulum enter 1, to select CartPole enter 0: '))
    if choice == 1:
        i = 2
        for j in range(1,5):
            path_exp_csv = exp_path + '\pendulum_exp_' + str(j) + str(i)
            filename = 'pendulum_exp_' + str(j) + str(i) + '.pdf'
            df_list = []
            for k in range(1, 4):
                df = pd.read_csv(path_exp_csv + '\pendulum_exp_' + str(j) + str(i) + '_' + str(k) + '.csv', sep='\t')
                df = df.rename(columns={'avg_reward':'avg_reward_' + str(k), 'std_reward':'std_reward_' + str(k)})
                df_list.append(df)
            full_df = pd.concat(df_list, axis=1)
            full_df['avg'] = full_df[['avg_reward_1', 'avg_reward_2', 'avg_reward_3']].mean(axis=1)
            plot_and_save(full_df, filename)

    else:
        for i in range(1,3):
            for j in range(1,5):
                path_exp_csv = exp_path + '\cartpole_exp_' + str(j) + str(i)
                filename = 'cartpole_exp_' + str(j) + str(i) + '.pdf'
                df_list = []
                for k in range(1, 4):
                    df = pd.read_csv(path_exp_csv + '\cartpole_exp_' + str(j) + str(i) + '_' + str(k) + '.csv', sep='\t')
                    df = df.rename(columns={'avg_reward':'avg_reward_' + str(k), 'std_reward':'std_reward_' + str(k)})
                    df_list.append(df)
                full_df = pd.concat(df_list, axis=1)
                full_df['avg'] = full_df[['avg_reward_1', 'avg_reward_2', 'avg_reward_3']].mean(axis=1)
                plot_and_save(full_df, filename)