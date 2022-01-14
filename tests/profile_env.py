import os

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
import hydra
import matplotlib.pyplot as plt

from nocturne_utils.wrappers import create_env

@hydra.main(config_path='../cfgs/', config_name='config_test')
def main(cfg):
    cfg.scenario_path = '/Users/nathan/Desktop/projects/nocturne/scenarios/twenty_car_intersection.json'
    cfg.subscriber.ego_subscriber.include_goal_img = True

    print(cfg)
    print("Working directory : {}".format(os.getcwd()))

    step_time = 0
    n_steps = 100

    env = create_env(cfg)
    obs = env.reset()
    for _ in range(n_steps):
        t0 = time.time()
        obs, rew, done, info = env.step({})
        step_time += time.time() - t0

    print(f'Time per step: {round(step_time / n_steps, 5)}s, average FPS {round(n_steps / step_time, 2)}')

if __name__ == '__main__':
    main()
