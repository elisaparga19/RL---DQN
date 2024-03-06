import argparse

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # experiment arguments
    parser.add_argument('--env', type = str, help = 'environment name to run the experiment')
    parser.add_argument('--nb_rollouts', type = int, help = 'number of rollouts')
    parser.add_argument('--alpha', type = float, help = 'alpha value')
    parser.add_argument('--filename', type=str, help='path where the files will be registered')
    
    # consolidate args
    args = parser.parse_args()
    
    return args