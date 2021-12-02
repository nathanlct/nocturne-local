from pathlib import Path
import os

import hydra
import gym
import numpy as np
from rlutil.logging import log_utils
import wandb

from nocturne_utils.wrappers import create_goal_env

import rlutil.torch as torch
# torch.backends.cudnn.benchmark = True
import rlutil.torch.pytorch_util as ptu


# Algo
from algos.gcsl import buffer, gcsl, variants, networks

os.environ["DISPLAY"] = ":0.0"


@hydra.main(config_path='../../cfgs/', config_name='config')
def main(cfg):
    # setup wandb
    # logdir = Path(os.getcwd())
    if cfg.wandb_id is not None:
        wandb_id = cfg.wandb_id
    else:
        wandb_id = wandb.util.generate_id()
        # with open(os.path.join(logdir, 'wandb_id.txt'), 'w+') as f:
        #     f.write(wandb_id)
    wandb_mode = "disabled" if (cfg.debug or not cfg.wandb) else "online"

    if cfg.wandb:
        run = wandb.init(config=cfg,
                        project=cfg.wandb_name,
                        name=wandb_id,
                        group=cfg.experiment,
                        resume="allow",
                        settings=wandb.Settings(start_method="fork"),
                        mode=wandb_mode)

    gpu = False
    if 'cuda' in cfg.device:
        gpu = True
    ptu.set_gpu(gpu)
    if not gpu:
        print('Not using GPU. Will be slow.')

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = create_goal_env(cfg)
    env_params = dict(eval_freq=cfg.algo.eval_freq,
                      eval_episodes=cfg.algo.eval_episodes,
                      max_trajectory_length=cfg.algo.max_trajectory_length,
                      max_timesteps=cfg.algo.max_timesteps,
                      # warning, use odd, not even, granularities so that zero is included
                      action_granularity=cfg.algo.action_granularity,
                      expl_noise=cfg.algo.expl_noise,
                      goal_threshold=cfg.rew_cfg.goal_tolerance,
                      support_termination=cfg.algo.support_termination,
                      go_explore=cfg.algo.go_explore,
                      save_video=cfg.algo.save_video,
                      wandb=cfg.algo.wandb,
                      explore_timesteps=cfg.algo.explore_timesteps,
                      batch_size=cfg.algo.batch_size,
                      buffer_size=cfg.algo.buffer_size)
    print(env_params)

    env, policy, replay_buffer, gcsl_kwargs = variants.get_params(
        env, env_params)
    algo = gcsl.GCSL(env, policy, replay_buffer, **gcsl_kwargs)

    exp_prefix = 'example/%s/gcsl/' % ('intersection', )

    # TODO(eugenevinitsky) logdir
    with log_utils.setup_logger(exp_prefix=exp_prefix, log_base_dir='./'):
        algo.train()

    if cfg.wandb:
        run.finish()


if __name__ == '__main__':
    main()
