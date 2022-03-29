import os

import hydra
import matplotlib.pyplot as plt

from envs.base_env import BaseEnv
from nocturne_utils.wrappers import create_env
from hydra import compose, initialize

os.environ["DISPLAY"] = ":0.0"


def test_rl_env():
    initialize(config_path="../cfgs/")
    cfg = compose(config_name="config")

    cfg.scenario_path = os.path.join(os.getcwd(), 'tests')
    cfg.subscriber.ego_subscriber.include_goal_img = True
    env = create_env(cfg)
    obs = env.reset()
    for _ in range(10):
        obs, rew, done, info = env.step({8: {'accel': 2.0, 'turn': 1.0}})


if __name__ == '__main__':
    test_rl_env()
