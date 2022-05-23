"""Utils that we use to understand the datasets we are working with."""
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROJECT_PATH
from nocturne import Simulation


def run_speed_test(files):
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
        sim = Simulation(os.path.join(PROCESSED_TRAIN_NO_TL, file), 0, False)
        vehs = sim.scenario().getObjectsThatMoved()
        scenario = sim.getScenario()
        for veh in vehs:
            t = time.perf_counter()
            obs = scenario.flattened_visible_state(veh, 120, np.pi)
            times_list.append(time.perf_counter() - t)
    print('avg, std. time to get obs is {}, {}'.format(np.mean(times_list),
                                                       np.std(times_list)))


def analyze_accels():
    """Plot the expert accels and number of observed moving vehicles."""
    f_path = PROCESSED_TRAIN_NO_TL
    with open(os.path.join(f_path, 'valid_files.json')) as file:
        valid_veh_dict = json.load(file)
        files = list(valid_veh_dict.keys())
    run_speed_test(files[0:10])


if __name__ == '__main__':
    analyze_accels()
