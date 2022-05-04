import os

from hydra import compose, initialize
from pathlib import Path

from cfgs.config import PROJECT_PATH
from nocturne_utils.wrappers import create_env

os.environ["DISPLAY"] = ":0.0"


def test_rl_env():
    initialize(config_path="../cfgs/")
    cfg = compose(config_name="config")
    # move the scenario path to this folder
    cfg.scenario_path = os.path.join(PROJECT_PATH, 'tests')
    cfg.max_num_vehicles = 50
    env = create_env(cfg)
    env.files = [str(PROJECT_PATH / "tests/large_file.json")]
    obs = env.reset()
    # quick check that rendering works
    img = env.scenario.getCone(env.scenario.getVehicles()[0], 120.0,
                               1.99 * 3.14, 0.0, False)
    img = env.render()
    for _ in range(10):
        obs, rew, done, info = env.step({8: {'accel': 2.0, 'turn': 1.0}})


if __name__ == '__main__':
    test_rl_env()
