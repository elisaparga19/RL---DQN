for i in 1 2 3
do
    env='CartPole-v1'
    filename_1=cartpole_exp_11_${i}
    filename_2=cartpole_exp_21_${i}
    filename_3=cartpole_exp_31_${i}
    python train_agent.py --env $env --planning_horizon 1 --nb_trajectories 200 --filename $filename_1
    python train_agent.py --env $env --planning_horizon 5 --nb_trajectories 200 --filename $filename_2
    python train_agent.py --env $env --planning_horizon 15 --nb_trajectories 200 --filename $filename_3
done

for i in 1 2 3
do
    env='Pendulum-v1'
    filename_1=pendulum_exp_11_${i}
    filename_2=pendulum_exp_21_${i}
    filename_3=pendulum_exp_31_${i}
    python train_agent.py --env $env --planning_horizon 30 --nb_trajectories 100 --filename $filename_1
    python train_agent.py --env $env --planning_horizon 30 --nb_trajectories 500 --filename $filename_2
    python train_agent.py --env $env --planning_horizon 30 --nb_trajectories 1000 --filename $filename_3
done