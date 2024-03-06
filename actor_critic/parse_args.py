import argparse

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # experiment arguments
    parser.add_argument('--env', type = str, help = 'environment name to run the experiment')
    parser.add_argument('--training_iterations', type = int, help = 'number of iterations')
    parser.add_argument('--batch_size', type = int, help = 'maximum number of elements in the batch')
    parser.add_argument('--nb_critic_updates', type = int, help = 'number of updates in critic network')
    parser.add_argument('--critic_lr', type = float, help = 'learning rate critic network')
    parser.add_argument('--filename', type=str, help='path where the files will be registered')
    
    # consolidate args
    args = parser.parse_args()
    
    return args