"""Make a movie from a random file."""
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from pyvirtualdisplay import Display

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROCESSED_VALID_NO_TL
from nocturne import Simulation

disp = Display()
disp.start()
fig = plt.figure()
files = os.listdir(PROCESSED_TRAIN_NO_TL)
# file = os.path.join(PROCESSED_TRAIN_NO_TL,
#                     files[np.random.randint(len(files))])
file = os.path.join(PROCESSED_VALID_NO_TL, 'tfrecord-00080-of-00150_60.json')
sim = Simulation(file, start_time=0)
frames = []
scenario = sim.getScenario()
for veh in scenario.getVehicles():
    veh.expert_control = True
for i in range(90):
    img = scenario.getImage(
        img_width=1600,
        img_height=1600,
        draw_target_positions=False,
        padding=50.0,
    )
    frames.append(img)
    sim.step(0.1)

movie_frames = np.array(frames)
output_path = f'{os.path.basename(file)}.mp4'
imageio.mimwrite(output_path, movie_frames, fps=30)
print('>', output_path)
