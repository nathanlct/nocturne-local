import os

import hydra

from envs.base_env import BaseEnv
from utils.wrappers import create_env

os.environ["DISPLAY"] = ":0.0"

@hydra.main(config_path='../cfgs/', config_name='config')
def main(cfg):
    env = create_env(cfg)
    obs = env.reset()
    print(obs)
    for i in range(10):
        obs, rew, done, info = env.step({8: {'accel': 2.0, 'turn': 1.0}})
        print(obs)

if __name__ == '__main__':
    main()