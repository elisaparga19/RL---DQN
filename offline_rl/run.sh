for i in 1 2 3
do
    env='CartPole-v0'
    filename_1=cartpole_exp_11_${i}
    filename_2=cartpole_exp_21_${i}
    filename_3=cartpole_exp_31_${i}
    python3 train_agent.py --env $env --nb_rollouts 10 --alpha 0.0 --filename $filename_1
    python3 train_agent.py --env $env --nb_rollouts 100 --alpha 0.0 --filename $filename_2
    python3 train_agent.py --env $env --nb_rollouts 1000 --alpha 0.0 --filename $filename_3

    filename_4=cartpole_exp_12_${i}
    filename_5=cartpole_exp_22_${i}
    filename_6=cartpole_exp_32_${i}
    python3 train_agent.py --env $env --nb_rollouts 10 --alpha 5.0 --filename $filename_4
    python3 train_agent.py --env $env --nb_rollouts 100 --alpha 5.0 --filename $filename_5
    python3 train_agent.py --env $env --nb_rollouts 1000 --alpha 5.0 --filename $filename_6

    filename_7=cartpole_exp_13_${i}
    filename_8=cartpole_exp_23_${i}
    filename_9=cartpole_exp_33_${i}
    python3 train_agent.py --env $env --nb_rollouts 10 --alpha 10.0 --filename $filename_7
    python3 train_agent.py --env $env --nb_rollouts 100 --alpha 10.0 --filename $filename_8
    python3 train_agent.py --env $env --nb_rollouts 1000 --alpha 10.0 --filename $filename_9

    filename_10=cartpole_exp_14_${i}
    filename_11=cartpole_exp_24_${i}
    filename_12=cartpole_exp_34_${i}
    python3 train_agent.py --env $env --nb_rollouts 10 --alpha 20.0 --filename $filename_10
    python3 train_agent.py --env $env --nb_rollouts 100 --alpha 20.0 --filename $filename_11
    python3 train_agent.py --env $env --nb_rollouts 1000 --alpha 20.0 --filename $filename_12
done

for i in 1 2 3
do
    env='MountainCar-v0'
    filename_1=mountaincar_exp_11_${i}
    filename_2=mountaincar_exp_21_${i}
    filename_3=mountaincar_exp_31_${i}
    python3 train_agent.py --env $env --nb_rollouts 10 --alpha 0.0 --filename $filename_1
    python3 train_agent.py --env $env --nb_rollouts 100 --alpha 0.0 --filename $filename_2
    python3 train_agent.py --env $env --nb_rollouts 1000 --alpha 0.0 --filename $filename_3

    filename_4=mountaincar_exp_12_${i}
    filename_5=mountaincar_exp_22_${i}
    filename_6=mountaincar_exp_32_${i}
    python3 train_agent.py --env $env --nb_rollouts 10 --alpha 5.0 --filename $filename_4
    python3 train_agent.py --env $env --nb_rollouts 100 --alpha 5.0 --filename $filename_5
    python3 train_agent.py --env $env --nb_rollouts 1000 --alpha 5.0 --filename $filename_6

    filename_7=mountaincar_exp_13_${i}
    filename_8=mountaincar_exp_23_${i}
    filename_9=mountaincar_exp_33_${i}
    python3 train_agent.py --env $env --nb_rollouts 10 --alpha 10.0 --filename $filename_7
    python3 train_agent.py --env $env --nb_rollouts 100 --alpha 10.0 --filename $filename_8
    python3 train_agent.py --env $env --nb_rollouts 1000 --alpha 10.0 --filename $filename_9

    filename_10=mountaincar_exp_14_${i}
    filename_11=mountaincar_exp_24_${i}
    filename_12=mountaincar_exp_34_${i}
    python3 train_agent.py --env $env --nb_rollouts 10 --alpha 20.0 --filename $filename_10
    python3 train_agent.py --env $env --nb_rollouts 100 --alpha 20.0 --filename $filename_11
    python3 train_agent.py --env $env --nb_rollouts 1000 --alpha 20.0 --filename $filename_12
done