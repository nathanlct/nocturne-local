import os

from hydra import compose, initialize
from pathlib import Path
from pyvirtualdisplay import Display

from cfgs.config import PROJECT_PATH
from nocturne_utils.wrappers import create_env

os.environ["DISPLAY"] = ":0.0"


def test_rl_env():
    disp = Display()
    disp.start()
    initialize(config_path="../cfgs/")
    cfg = compose(config_name="config")
    cfg.max_num_vehicles = 50
    env = create_env(cfg)
    env.files = [str(PROJECT_PATH / "tests/large_file.json")]
    obs = env.reset()
    # quick check that rendering works
    img = env.scenario.getCone(env.scenario.getVehicles()[0], 120.0,
                               1.99 * 3.14, 0.0, False)
    veh = env.scenario.getVehicles()[1]
    veh_id = veh.getID()
    for _ in range(10):
        prev_position = [veh.position.x, veh.position.y]
        obs, rew, done, info = env.step({veh_id: {'accel': 2.0, 'turn': 1.0}})
        if veh not in env.scenario.getVehicles():
            break
        new_position = [veh.position.x, veh.position.y]
        assert prev_position != new_position, f'veh was in position {prev_position} which is the same as {new_position} but should have moved'


if __name__ == '__main__':
    test_rl_env()
