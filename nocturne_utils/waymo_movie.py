"""Make a movie from a random file."""
import os

from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np
from pyvirtualdisplay import Display

from cfgs.config import PROCESSED_TRAIN_NO_TL
from nocturne import Simulation

disp = Display()
disp.start()
fig = plt.figure()
cam = Camera(fig)
files = os.listdir(PROCESSED_TRAIN_NO_TL)
file = os.path.join(PROCESSED_TRAIN_NO_TL,
                    files[np.random.randint(len(files))])
sim = Simulation(file, start_time=0)
scenario = sim.getScenario()
for veh in scenario.getVehicles():
    veh.expert_control = True
for i in range(90):
    img = scenario.getImage(None, render_goals=True)
    plt.imshow(img)
    cam.snap()
    sim.step(0.1)

animation = cam.animate(interval=50)
animation.save(f'{os.path.basename(file)}.mp4')
