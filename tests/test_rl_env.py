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
    for _ in range(90):
        vehs = env.scenario.getObjectsThatMoved()
        prev_position = {
            veh.getID(): [veh.position.x, veh.position.y]
            for veh in vehs
        }
        obs, rew, done, info = env.step(
            {veh.getID(): {
                'accel': 2.0,
                'turn': 1.0
            }
             for veh in vehs})
        for veh in vehs:
            if veh in env.scenario.getObjectsThatMoved():
                new_position = [veh.position.x, veh.position.y]
                assert prev_position[veh.getID(
                )] != new_position, f'veh {veh.getID()} was in position \
                    {prev_position[veh.getID()]} which is the \
                        same as {new_position} but should have moved'


if __name__ == '__main__':
    test_rl_env()
