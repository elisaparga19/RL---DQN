import argparse

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # experiment arguments
    parser.add_argument('--env', type = str, help = 'environment name to run the experiment')
    parser.add_argument('--planning_horizon', type = int, help = 'number of actions of a trajectory')
    parser.add_argument('--nb_trajectories', type = int, help = 'number of trajectories')
    parser.add_argument('--filename', type=str, help='path where the files will be registered')
    
    # consolidate args
    args = parser.parse_args()
    
    return args