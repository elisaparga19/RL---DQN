import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#exp_path = os.path.abspath('../A__T2')
exp_path = os.path.abspath('../Tarea 3/policy_gradients')

def plot_performance_metrics(axes, tr_steps_vec, avg_reward_vec, std_reward_vec, success_rate_vec, std_success_rate_vec, i, j):
    ax1, ax2 = axes
    
    [ax.cla() for ax in axes]
    ax1.errorbar(tr_steps_vec, avg_reward_vec, yerr=std_reward_vec, marker='.',color='C0')
    ax1.set_ylabel('Avg Reward')
    ax2.errorbar(tr_steps_vec, success_rate_vec, yerr=std_success_rate_vec, marker='.',color='C1')
    ax2.set_ylabel('Success Rate')

    [ax.grid('on') for ax in axes]
    [ax.set_xlabel('Training step') for ax in axes]
    plt.savefig('metrics_exp_' + str(i) + str(j) + '.pdf')
    plt.close()


for i in range(1,4):
    for j in range(1,4):
        path_exp_csv = exp_path + '\exp_' + str(i) + str(j)
        exp_files = os.listdir(path_exp_csv)
        df_list = []
        for k in range(len(exp_files)):
            df = pd.read_csv(path_exp_csv + '\exp_' + str(i) + '_' + str(j) + '_' + str(k) + '.csv', sep='\t')
            df_list.append(df)
        full_df = pd.concat(df_list)
        full_df = full_df.groupby(full_df.index).agg(avg_reward=('avg_reward', np.mean), 
                                                     avg_success_rate=('success_rate', np.mean), 
                                                     std_reward=('avg_reward', lambda x: np.std(x, ddof=0)), 
                                                     std_success_rate=('success_rate', lambda x: np.std(x, ddof=0)))
        _, (axes) = plt.subplots(1, 2, figsize=(12,4))
        plot_performance_metrics(axes, np.arange(0, 21, 1), 
                                 full_df.loc[:, 'avg_reward'].tolist(), 
                                 full_df.loc[:, 'std_reward'].tolist(), 
                                 full_df.loc[:, 'avg_success_rate'].tolist(), 
                                 full_df.loc[:, 'std_success_rate'].tolist(), 
                                 i, j)
