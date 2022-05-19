"""Utils that we use to understand the datasets we are working with."""
import os

import matplotlib.pyplot as plt
import numpy as np

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROJECT_PATH
from nocturne import Simulation


def run_analysis(files):
    """Compute the expert accelerations and number of vehicles across the dataset.

    Args:
        files ([str]): List of files to analyze

    Returns
    -------
        [np.float], [np.float]: List of expert accels, list of number
                                of moving vehicles in file
    """
    observed_accels = []
    num_vehicles = []
    for file_idx, file in enumerate(files):
        sim = Simulation(os.path.join(PROCESSED_TRAIN_NO_TL, file), 0, False)
        vehs = sim.scenario().getObjectsThatMoved()
        # this checks if the vehicles has actually moved any distance at all
        valid_vehs = []
        prev_speeds = []
        for veh in vehs:
            veh.expert_control = True
            obj_pos = veh.position
            goal_pos = veh.destination
            if (obj_pos - goal_pos).norm() > 0.5:
                valid_vehs.append(veh)
            if veh in valid_vehs:
                veh_speed = sim.scenario().getExpertSpeeds(0, veh.id)
                veh_speed = np.linalg.norm([veh_speed.x, veh_speed.y])
                if not np.isclose(veh.position.x, -10000.0):
                    prev_speeds.append(
                        (veh_speed, True, [veh.position.x, veh.position.y], 0))
                else:
                    prev_speeds.append(
                        (veh_speed, False, [veh.position.x,
                                            veh.position.y], 0))
        num_vehicles.append(len(valid_vehs))
        sim.step(0.1)
        for i in range(1, 90):
            for veh_index, veh in enumerate(valid_vehs):
                # check if the vehicle is actually valid
                veh_speed = sim.scenario().getExpertSpeeds(i, veh.id)
                veh_speed = veh_speed.norm()
                if np.isclose(veh.position.x, -10000.0):
                    prev_speeds[veh_index] = (veh_speed, False,
                                              [veh.position.x,
                                               veh.position.y], i)
                else:
                    # approximate the accel using an euler step but only
                    # if the prior step was a step where the agent
                    # was valid
                    if prev_speeds[veh_index][1]:
                        accel = (veh_speed - prev_speeds[veh_index][0]) / 0.1
                        observed_accels.append(accel)
                    prev_speeds[veh_index] = (veh_speed, True,
                                              [veh.position.x,
                                               veh.position.y], i)
            sim.step(0.1)

        if file_idx > 300:
            break
    return observed_accels, num_vehicles


def analyze_accels():
    """Plot the expert accels and number of observed moving vehicles."""
    f_path = PROCESSED_TRAIN_NO_TL
    with open(os.path.join(f_path, 'valid_files.txt')) as file:
        files = [line.strip() for line in file]
    observed_accels_valid, num_vehicles_valid = run_analysis(files)
    with open(os.path.join(f_path, 'invalid_files.txt')) as file:
        files = [line.strip() for line in file]
    _, num_vehicles_invalid = run_analysis(files)

    output_path = os.path.join(PROJECT_PATH, 'nocturne_utils/data_analysis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    observed_accels = np.array(observed_accels_valid)
    print(np.max(observed_accels))
    print(np.min(observed_accels))
    observed_accels = observed_accels[np.abs(observed_accels) < 5]
    plt.figure()
    plt.hist(observed_accels)
    plt.savefig(os.path.join(output_path, 'observed_accels.png'))
    plt.figure()
    plt.hist(
        num_vehicles_valid,
        bins=30,
        density=True,
        histtype='step',
        cumulative=True,
    )
    plt.hist(
        num_vehicles_invalid,
        bins=30,
        density=True,
        histtype='step',
        cumulative=True,
    )
    plt.legend(['valid', 'invalid'])
    plt.savefig(os.path.join(output_path, 'num_vehs_cdf.png'))
    plt.figure()
    plt.hist(num_vehicles_valid, bins=30, alpha=0.5, color='b')
    plt.axvline(np.mean(num_vehicles_valid), color='b', label='_nolegend_')
    plt.hist(num_vehicles_invalid, bins=30, alpha=0.5, color='r')
    plt.axvline(np.mean(num_vehicles_invalid), color='r', label='_nolegend_')
    plt.legend(['valid', 'invalid'])
    plt.savefig(os.path.join(output_path, 'num_vehs_hist.png'))


if __name__ == '__main__':
    analyze_accels()
