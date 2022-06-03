"""Utils that we use to understand the datasets we are working with."""
import json
import os
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
from pyvirtualdisplay import Display

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROJECT_PATH,  get_scenario_dict
from nocturne import Simulation, Action


def run_speed_test(files, cfg):
    """Compute the expert accelerations and number of vehicles across the dataset.

    Args:
        files ([str]): List of files to analyze

    Returns
    -------
        [np.float], [np.float]: List of expert accels, list of number
                                of moving vehicles in file
    """
    times_list = []
    for file_idx, file in enumerate(files):
        sim = Simulation(os.path.join(PROCESSED_TRAIN_NO_TL, file), get_scenario_dict(cfg))
        vehs = sim.scenario().getObjectsThatMoved()
        scenario = sim.getScenario()
        for veh in vehs:
            t = time.perf_counter()
            obs = scenario.flattened_visible_state(veh, 80,
                                                   (180 / 180) * np.pi)
            # scenario.getConeImage(
            #     source=veh,
            #     view_dist=120.0,
            #     view_angle=np.pi,
            #     head_tilt=0.0,
            #     img_width=400,
            #     img_height=400,
            #     padding=0.0,
            #     draw_target_position=True,
            # )
            #           veh.apply_action(Action(1.0, 1.0))

            #            sim.step(0.1)
            times_list.append(time.perf_counter() - t)
    print('avg, std. time to get obs is {}, {}'.format(np.mean(times_list),
                                                       np.std(times_list)))

@hydra.main(config_path="../../cfgs/", config_name="config")
def analyze_accels(cfg):
    """Plot the expert accels and number of observed moving vehicles."""
    f_path = PROCESSED_TRAIN_NO_TL
    with open(os.path.join(f_path, 'valid_files.json')) as file:
        valid_veh_dict = json.load(file)
        files = list(valid_veh_dict.keys())
    run_speed_test(files[0:10], cfg)


if __name__ == '__main__':
    disp = Display()
    disp.start()
    analyze_accels()
