# This file runs through the data to look for cases where there are undesirable corner cases
# the cases we currently check for are:
# 1) is a vehicle initialized in a colliding state with another vehicle
# 2) is a vehicle initialized in a colliding state with a road edge?
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

from cfgs.config import PROCESSED_DATA_PATH_NO_TL, PROJECT_PATH
from nocturne import Simulation

os.environ["DISPLAY"] = ":0.0"

if __name__ == '__main__':
    output_folder = 'corner_case_vis'
    output_path = Path(PROJECT_PATH) / f'nocturne_utils/{output_folder}'
    output_path.mkdir(exist_ok=True)
    files = list(os.listdir(PROCESSED_DATA_PATH_NO_TL))
    for file in files:
        print('about to load scenario')
        sim = Simulation(os.path.join(PROCESSED_DATA_PATH_NO_TL, file))
        print('loaded the scenario')
        vehs = sim.getScenario().getVehicles()
        print('grabbed the vehicles')
        for i, veh in enumerate(vehs):
            print('in vehicle loop')
            collided = veh.getCollided()
            print('called get collided')
            if collided:
                print('making an image')
                plt.figure()
                plt.imshow(sim.getScenario().getImage(None, render_goals=True))
                plt.title(f'{file}_{i}')
                plt.savefig(f'{output_folder}/{file}_{i}.png')
