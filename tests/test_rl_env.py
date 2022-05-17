"""Test step and rendering functions."""

from hydra import compose, initialize
from pyvirtualdisplay import Display

from cfgs.config import PROJECT_PATH
<<<<<<< HEAD
from nocturne.envs.wrappers import create_env
=======
from nocturne_utils.wrappers import create_env
>>>>>>> bc7f21b1d4342cdd17fbffb443eeb0d66d9479ce


def test_rl_env():
    """Test step and rendering functions."""
    disp = Display()
    disp.start()
    initialize(config_path="../cfgs/")
    cfg = compose(config_name="config")
    cfg.scenario_path = os.path.join(PROJECT_PATH, 'tests')
    cfg.max_num_vehicles = 50
    env = create_env(cfg)
    env.files = [str(PROJECT_PATH / "tests/large_file.json")]
    _ = env.reset()
    # quick check that rendering works
    _ = env.scenario.getCone(env.scenario.getVehicles()[0], 120.0, 1.99 * 3.14,
                             0.0, False)
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
