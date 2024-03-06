import argparse

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # experiment arguments
    parser.add_argument('--env', type = str, help = 'environment name to run the experiment')
    parser.add_argument('--training_iterations', type = int, help = 'number of iterations')
    parser.add_argument('--batch_size', type = int, help = 'maximum number of elements in the batch')
    parser.add_argument('--use_baseline', type = bool, help = 'whether to use a baseline or not')
    parser.add_argument('--reward_to_go', type = bool, help = 'whether to use a reward to go or not')
    parser.add_argument('--filename', type=str, help='path where the files will be registered')
    
    # consolidate args
    args = parser.parse_args()
    
    return args