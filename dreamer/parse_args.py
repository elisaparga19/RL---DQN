import argparse
from env import GYM_ENVS, CONTROL_SUITE_ENVS
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Dream')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    parser.add_argument('--env', type=str, default='walker-walk', choices=GYM_ENVS + CONTROL_SUITE_ENVS, help='Gym/Control Suite environment')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length') #hola
    parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
    parser.add_argument('--dense-act', type=str, default='elu', choices=dir(F), help='Model activation function a dense layer')
    parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
    parser.add_argument('--hidden-size', type=int, default=300, metavar='H', help='Hidden size')  # paper:300; tf_implementation:400; aligned wit paper. 
    parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
    parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
    parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')
    parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes') #hola
    parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes') # hola
    parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval') 
    parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
    parser.add_argument('--sequence-length', type=int, default=50, metavar='L', help='Sequence Length')
    parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
    parser.add_argument('--reward_scale', type=float, default=5.0, help='coefficiency term of reward loss')
    parser.add_argument('--pcont_scale', type=float, default=5.0, help='coefficiency term of pcont loss')
    parser.add_argument('--world_lr', type=float, default=6e-4, metavar='α', help='Learning rate') 
    parser.add_argument('--actor_lr', type=float, default=8e-5, metavar='α', help='Learning rate') 
    parser.add_argument('--value_lr', type=float, default=8e-5, metavar='α', help='Learning rate') 
    #parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)') 
    #parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value') 
    # Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
    parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')
    parser.add_argument('--expl_amount', type=float, default=0.3, help='exploration noise')
    parser.add_argument('--planning-horizon', type=int, default=15, metavar='H', help='Planning horizon distance')
    parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
    parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')
    parser.add_argument('--test', action='store_true', help='Test only') # hola
    parser.add_argument('--test-interval', type=int, default=25, metavar='I', help='Test interval (episodes)') # hola
    parser.add_argument('--test-episodes', type=int, default=10, metavar='E', help='Number of test episodes') # hola
    parser.add_argument('--checkpoint-interval', type=int, default=100, metavar='I', help='Checkpoint interval (episodes)') # hola
    parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay') # hola
    # parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
    # parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--pcont', action='store_true', help="use the pcont to predict the continuity")
    parser.add_argument('--with_logprob', action='store_true', help='use the entropy regularization')

    args = parser.parse_args()
    
    return args