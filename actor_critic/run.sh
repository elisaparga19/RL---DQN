for i in 1 2 3
do
    env='CartPole-v1'

    filename_1=cartpole_exp_11_${i}
    filename_2=cartpole_exp_21_${i}
    filename_3=cartpole_exp_31_${i}
    
    filename_4=cartpole_exp_12_${i}
    filename_5=cartpole_exp_22_${i}
    filename_6=cartpole_exp_32_${i}

    python train_agent.py --env $env --training_iterations 200 --batch_size 500 --nb_critic_updates 1 --critic_lr 0.001 --filename $filename_1
    python train_agent.py --env $env --training_iterations 200 --batch_size 500 --nb_critic_updates 10 --critic_lr 0.001 --filename $filename_2
    python train_agent.py --env $env --training_iterations 200 --batch_size 500 --nb_critic_updates 100 --critic_lr 0.001 --filename $filename_3

    python train_agent.py --env $env --training_iterations 200 --batch_size 5000 --nb_critic_updates 1 --critic_lr 0.001 --filename $filename_4
    python train_agent.py --env $env --training_iterations 200 --batch_size 5000 --nb_critic_updates 10 --critic_lr 0.001 --filename $filename_5
    python train_agent.py --env $env --training_iterations 200 --batch_size 5000 --nb_critic_updates 100 --critic_lr 0.001 --filename $filename_6
done

for i in 1 2 3
do
    env='Acrobot-v1'

    filename_1=acrobot_exp_11_${i}
    filename_2=acrobot_exp_21_${i}
    filename_3=acrobot_exp_31_${i}
    
    filename_4=acrobot_exp_12_${i}
    filename_5=acrobot_exp_22_${i}
    filename_6=acrobot_exp_32_${i}

    python train_agent.py --env $env --training_iterations 200 --batch_size 500 --nb_critic_updates 1 --critic_lr 0.001 --filename $filename_1
    python train_agent.py --env $env --training_iterations 200 --batch_size 500 --nb_critic_updates 10 --critic_lr 0.001 --filename $filename_2
    python train_agent.py --env $env --training_iterations 200 --batch_size 500 --nb_critic_updates 100 --critic_lr 0.001 --filename $filename_3

    python train_agent.py --env $env --training_iterations 200 --batch_size 5000 --nb_critic_updates 1 --critic_lr 0.001 --filename $filename_4
    python train_agent.py --env $env --training_iterations 200 --batch_size 5000 --nb_critic_updates 10 --critic_lr 0.001 --filename $filename_5
    python train_agent.py --env $env --training_iterations 200 --batch_size 5000 --nb_critic_updates 100 --critic_lr 0.001 --filename $filename_6
done

for i in 1 2 3
do
    env='Pendulum-v1'

    filename_1=pendulum_exp_12_${i}
    filename_2=pendulum_exp_22_${i}
    filename_3=pendulum_exp_32_${i}

    python train_agent.py --env $env --training_iterations 2000 --batch_size 5000 --nb_critic_updates 1 --critic_lr 0.001 --filename $filename_1
    python train_agent.py --env $env --training_iterations 2000 --batch_size 5000 --nb_critic_updates 10 --critic_lr 0.001 --filename $filename_2
    python train_agent.py --env $env --training_iterations 2000 --batch_size 5000 --nb_critic_updates 100 --critic_lr 0.001 --filename $filename_3
done

for i in 1 2 3
do
    env='Pendulum-v1'

    filename_4=pendulum_exp_12_critic_${i}
    filename_5=pendulum_exp_22_critic_${i}
    filename_6=pendulum_exp_32_critic_${i}

    python train_agent.py --env $env --training_iterations 2000 --batch_size 5000 --nb_critic_updates 1 --critic_lr 0.01 --filename $filename_4
    python train_agent.py --env $env --training_iterations 2000 --batch_size 5000 --nb_critic_updates 10 --critic_lr 0.01 --filename $filename_5
    python train_agent.py --env $env --training_iterations 2000 --batch_size 5000 --nb_critic_updates 100 --critic_lr 0.01 --filename $filename_6
done