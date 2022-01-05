import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

from envs.base_env import BaseEnv
from nocturne_utils.wrappers import create_env

from pyvirtualdisplay import Display

@hydra.main(config_path='../cfgs/', config_name='config')
def main(cfg):
    disp = Display()
    disp.start()

    cfg.scenario_path = '/private/home/eugenevinitsky/Code/nocturne/scenarios/test_intersection.json'
    # remove for speed
    cfg.subscriber.ego_subscriber.include_goal_img = False
    # this is needed for the tests to pass
    cfg.subscriber.use_local_coordinates = True
    env = create_env(cfg)
    test_road_objects(env)

def test_road_objects(env):
    # the two agents are positioned symmetrically so first off, we know that their road objects
    # should be the same
    obs = env.reset()
    agent_1 = obs[8]
    agent_2 = obs[9]
    npt.assert_allclose(agent_1['road_objects'], agent_2['road_objects'], rtol=1e-2)

    # next, we know that agent 1 is located at [20, -200] so lets use that to compute all the 
    # correct distances and headings
    expected_distances = [22.0]
    vehicle_obj = env.vehicles[0]
    for i, road_obj in enumerate(env.scenario.getRoadObjects()):
        # don't check against vehicles
        if road_obj.getType() != 'Object':
            continue
        new_obs = env.subscriber.road_subscriber.get_obs(road_obj, vehicle_obj)

def create_env_from_cfg(cfg):
    return create_env(cfg)

if __name__ == '__main__':
    main()
