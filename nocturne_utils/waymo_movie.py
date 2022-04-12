# TODO(ev) make this efficient and called by step rather than re-rendering the scene
import os

from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np
from pyvirtualdisplay import Display

from nocturne import Simulation

disp = Display()
disp.start()
fig = plt.figure()
cam = Camera(fig)
path = '/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/uncompressed/scenario/formatted_json'
files = os.listdir(path)
file = os.path.join(path, files[np.random.randint(len(files))])
sim = Simulation(file, start_time=0)
scenario = sim.getScenario()
for veh in scenario.getVehicles():
    veh.set_expert_controlled(True)
for i in range(90):
    img = np.array(scenario.getImage(None, render_goals=True), copy=False)
    plt.imshow(img)
    cam.snap()
    sim.step(0.1)

animation = cam.animate(interval=50)
animation.save(f'{os.path.basename(file)}.mp4')