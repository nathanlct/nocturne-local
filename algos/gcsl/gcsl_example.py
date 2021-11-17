import os

import hydra
import gym
import numpy as np
from rlutil.logging import log_utils, logger

from nocturne_utils.wrappers import create_goal_env

import rlutil.torch as torch
import rlutil.torch.pytorch_util as ptu

# Envs
from algos.gcsl.env_utils import DiscretizedActionEnv

# Algo
from algos.gcsl import buffer, gcsl, variants, networks

os.environ["DISPLAY"] = ":0.0"


@hydra.main(config_path='../../cfgs/', config_name='config')
def main(cfg):
    gpu = False
    if 'cuda' in cfg.device:
        gpu = True
    ptu.set_gpu(gpu)
    if not gpu:
        print('Not using GPU. Will be slow.')

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = create_goal_env(cfg)
    env_params = dict(eval_freq=2000,
                      eval_episodes=2,
                      max_trajectory_length=50,
                      max_timesteps=1e6,
                      # warning, use odd, not even, granularities so that zero is included
                      action_granularity=5,
                      expl_noise=0.1,
                      goal_threshold=cfg.rew_cfg.goal_tolerance,
                      support_termination=True,
                      go_explore=True,
                      save_video=True)
    print(env_params)

    env, policy, replay_buffer, gcsl_kwargs = variants.get_params(
        env, env_params)
    algo = gcsl.GCSL(env, policy, replay_buffer, **gcsl_kwargs)

    exp_prefix = 'example/%s/gcsl/' % ('intersection', )

    # TODO(eugenevinitsky) logdir
    with log_utils.setup_logger(exp_prefix=exp_prefix, log_base_dir='./'):
        algo.train()


if __name__ == '__main__':
    main()
