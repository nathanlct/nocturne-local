"""Test step and rendering functions."""
import os

from hydra import compose, initialize
import numpy as np
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
    env = create_env(cfg)
    env.files = [str(PROJECT_PATH / "tests/large_file.json")]
    _ = env.reset()
    # quick check that rendering works
    _ = env.scenario.getConeImage(
        source=env.scenario.getObjectsThatMoved()[0],
        view_dist=120.0,
        view_angle=np.pi * 0.8,
        head_tilt=0.0,
        img_width=1600,
        img_height=1600,
        padding=50.0,
        draw_target_position=True,
    )
    for _ in range(90):
        vehs = env.scenario.getObjectsThatMoved()
        prev_position = {
            veh.getID(): [veh.position.x, veh.position.y]
            for veh in vehs
        }
        obs, rew, done, info = env.step(
            {veh.id: Action(acceleration=2.0, steering=1.0)
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
