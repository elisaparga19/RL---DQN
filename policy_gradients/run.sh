for i in 1 2 3
do
    env='CartPole-v1'
    filename_1=cartpole_exp_11_${i}
    filename_2=cartpole_exp_21_${i}
    filename_3=cartpole_exp_31_${i}
    filename_4=cartpole_exp_41_${i}
    python train_agent.py --env $env --training_iterations 100 --batch_size 500 --use_baseline False --reward_to_go False --filename $filename_1
    python train_agent.py --env $env --training_iterations 100 --batch_size 500 --use_baseline False --reward_to_go True --filename $filename_2
    python train_agent.py --env $env --training_iterations 100 --batch_size 500 --use_baseline True --reward_to_go False --filename $filename_3
    python train_agent.py --env $env --training_iterations 100 --batch_size 500 --use_baseline True --reward_to_go True --filename $filename_4
done

for i in 1 2 3
do
    env='CartPole-v1'
    filename_1=cartpole_exp_12_${i}
    filename_2=cartpole_exp_22_${i}
    filename_3=cartpole_exp_32_${i}
    filename_4=cartpole_exp_42_${i}
    python train_agent.py --env $env --training_iterations 100 --batch_size 5000 --use_baseline False --reward_to_go False --filename $filename_1
    python train_agent.py --env $env --training_iterations 100 --batch_size 5000 --use_baseline False --reward_to_go True --filename $filename_2
    python train_agent.py --env $env --training_iterations 100 --batch_size 5000 --use_baseline True --reward_to_go False --filename $filename_3
    python train_agent.py --env $env --training_iterations 100 --batch_size 5000 --use_baseline True --reward_to_go True --filename $filename_4
done

for i in 1 2 3
do
    env='Pendulum-v1'
    filename_1=pendulum_exp_12_${i}
    filename_2=pendulum_exp_22_${i}
    filename_3=pendulum_exp_32_${i}
    filename_4=pendulum_exp_42_${i}
    python train_agent.py --env $env --training_iterations 1000 --batch_size 5000 --use_baseline False --reward_to_go False --filename $filename_1
    python train_agent.py --env $env --training_iterations 1000 --batch_size 5000 --use_baseline False --reward_to_go True --filename $filename_2
    python train_agent.py --env $env --training_iterations 1000 --batch_size 5000 --use_baseline True --reward_to_go False --filename $filename_3
    python train_agent.py --env $env --training_iterations 1000 --batch_size 5000 --use_baseline True --reward_to_go True --filename $filename_4
done

    