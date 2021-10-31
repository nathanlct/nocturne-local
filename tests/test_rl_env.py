import os

import hydra

from envs.base_env import BaseEnv

os.environ["DISPLAY"] = ":0.0"

@hydra.main(config_path='../cfgs/', config_name='config')
def main(cfg):
    env = BaseEnv(cfg)
    env.reset()
    import ipdb; ipdb.set_trace()
    env.step({'8': {'accel': 2.0, 'turn': 1.0}})

if __name__ == '__main__':
    main()