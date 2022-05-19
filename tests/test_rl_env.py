"""Test step and rendering functions."""
import os

from hydra import compose, initialize
from pyvirtualdisplay import Display

from cfgs.config import PROJECT_PATH
from nocturne import Action
from nocturne.envs.wrappers import create_env


def test_rl_env():
    """Test step and rendering functions."""
    disp = Display()
    disp.start()
    initialize(config_path="../cfgs/")
    cfg = compose(config_name="config")
    cfg.scenario_path = os.path.join(PROJECT_PATH, 'tests')
    cfg.max_num_vehicles = 50
    print(f"cfg = {cfg}")
    env = create_env(cfg)
    print("Create env finished")
    env.files = [str(PROJECT_PATH / "tests/large_file.json")]
    _ = env.reset()
    print("Reset finished")
    # quick check that rendering works
    _ = env.scenario.getConeImage(env.scenario.getVehicles()[0], 120.0,
                                  1.99 * 3.14, 0.0, False)
    # for _ in range(90):
    for step in range(90):
        print(f"step = {step}")
        vehs = env.scenario.getObjectsThatMoved()
        print(f"vehcile = {vehicle.id}")
        prev_position = {
            veh.getID(): [veh.position.x, veh.position.y]
            for veh in vehs
        }
        obs, rew, done, info = env.step(
            {veh.id: Action(2.0, 1.0)
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
