import os

import hydra

from envs.base_env import BaseEnv
from utils.wrappers import create_env

os.environ["DISPLAY"] = ":0.0"

@hydra.main(config_path='../cfgs/', config_name='config')
def main(cfg):
    env = create_env(cfg)
    env.reset()
    env.step({'8': {'accel': 2.0, 'turn': 1.0}})

if __name__ == '__main__':
    main()