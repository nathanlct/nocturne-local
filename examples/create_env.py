"""Test step and rendering functions."""
import hydra
from pyvirtualdisplay import Display

from nocturne import Action
from nocturne.envs.wrappers import create_env


@hydra.main(config_path="../cfgs/", config_name="config")
def create_rl_env(cfg):
    """Test step and rendering functions."""
    disp = Display()
    disp.start()
    env = create_env(cfg)
    _ = env.reset()
    # quick check that rendering works
    _ = env.scenario.getConeImage(env.scenario.getVehicles()[0], 120.0,
                                  1.99 * 3.14, 0.0, draw_target_position=False)
    for _ in range(80):
        vehs = env.scenario.getObjectsThatMoved()
        _ = {
            veh.getID(): [veh.position.x, veh.position.y]
            for veh in vehs
        }
        # obs, rew, done, info
        _, _, _, _ = env.step(
            {veh.id: Action(acceleration=2.0, steering=1.0)
             for veh in vehs})


if __name__ == '__main__':
    create_rl_env()
