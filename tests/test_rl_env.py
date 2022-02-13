import os

import hydra
import matplotlib.pyplot as plt

from envs.base_env import BaseEnv
from nocturne_utils.wrappers import create_env

from pyvirtualdisplay import Display

@hydra.main(config_path='../cfgs/', config_name='config')
def main(cfg):
    disp = Display()
    disp.start()

    cfg.scenario_path = '/private/home/eugenevinitsky/Code/nocturne/scenarios/two_car_intersection.json'
    cfg.subscriber.ego_subscriber.include_goal_img = True
    env = create_env(cfg)
    obs = env.reset()
    for _ in range(10):
        obs, rew, done, info = env.step({8: {'accel': 2.0, 'turn': 1.0}})
    goal_img = obs[9]['goal_img']
    plt.figure()
    plt.imshow(goal_img)
    plt.savefig('./goal_img.png')

if __name__ == '__main__':
    main()
