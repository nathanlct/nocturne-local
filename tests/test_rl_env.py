import os

from hydra import compose, initialize
from pathlib import Path

from cfgs.config import PROJECT_PATH
from nocturne_utils.wrappers import create_env

os.environ["DISPLAY"] = ":0.0"


def test_rl_env():
    initialize(config_path="../cfgs/")
    cfg = compose(config_name="config")
    cfg.subscriber.ego_subscriber.include_goal_img = True
    env = create_env(cfg)
    env.file = PROJECT_PATH / "tests/large_file.json"
    obs = env.reset()
    for _ in range(10):
        obs, rew, done, info = env.step({8: {'accel': 2.0, 'turn': 1.0}})


if __name__ == '__main__':
    test_rl_env()
