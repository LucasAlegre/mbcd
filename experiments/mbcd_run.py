import os
import gym
import numpy as np
import pandas as pd
import random 
import argparse

import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines.common.vec_env import VecNormalize
#from stable_baselines.common.policies import register_policy
from stable_baselines.sac.policies import FeedForwardPolicy

from mbcd.mbcd import MBCD
from mbcd.envs.non_stationary_wrapper import NonStationaryEnv

from mbcd.utils.util import evaluate
from mbcd.sac_mbcd import SAC


class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs, layers=[256, 256], feature_extraction="mlp")  # 64,64
#register_policy('CustomSACPolicy', CustomSACPolicy)

parser = argparse.ArgumentParser()
parser.add_argument('-seed', dest='seed', required=False, type=int, help="Seed\n", default=0)      
parser.add_argument('-algo', dest='algo', required=True, type=str, help="Algo [mbcd, mbpo, sac]\n")
parser.add_argument('-env', dest='env', required=False, type=str, help="Env [halfcheetah, pusher]", default='halfcheetah')                
args = parser.parse_args()
assert args.algo in ['mbcd', 'mbpo', 'sac']
mbcd = args.algo == 'mbcd'
mbpo = args.algo != 'sac'

SEED = args.seed
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)


def main(config):

    model = SAC(CustomSACPolicy,
                env=config['env'],
                rollout_schedule=config['rollout_schedule'],
                verbose=1,
                batch_size=config['batch_size'],
                gradient_steps=config['gradient_steps'],
                target_entropy='auto',
                ent_coef='auto',
                mbpo=mbpo,
                mbcd=mbcd,
                max_std=config['max_std'],
                num_stds=config['num_stds'],
                n_hidden_units_dynamics=config['n_hidden_units_dynamics'],
                n_layers_dynamics=config['n_layers_dynamics'],
                dynamics_memory_size=config['dynamics_memory_size'],
                cusum_threshold=config['cusum_threshold'],
                run_id=config['run_id'],
                tensorboard_log='./logs/',
                seed=SEED)

    model.learn(total_timesteps=config['total_timesteps'], tb_log_name='mbcd-test')
    if args.algo == 'sac':
        model.save('weights/'+'sacfinalpolicy')
    else:
        model.deepRLCD.save_current()
        model.deepRLCD.save_models()

if __name__ == '__main__':

    if args.env == 'halfcheetah':
        config = {
                'env': NonStationaryEnv(gym.envs.make('HalfCheetah-v2'), change_freq=40000),
                'rollout_schedule': [20000,50000,1,1],
                'batch_size': 256,
                'gradient_steps': 20,
                'target_entropy': 'auto',
                'ent_coef': 'auto',
                'max_std': 0.5,
                'num_stds': 2.0, 
                'n_hidden_units_dynamics': 200,
                'n_layers_dynamics': 4,
                'dynamics_memory_size': 100000,
                'cusum_threshold': 100,
                'run_id':'{}-halfcheetah-ns-paper{}'.format(args.algo, str(SEED)),
                'total_timesteps': 480000
        }


    main(config)



   


