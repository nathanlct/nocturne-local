# TODO(ev) make this efficient and called by step rather than re-rendering the scene
import os

from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np

from nocturne import Simulation
os.environ["DISPLAY"] = ":0.0"

fig = plt.figure()
cam = Camera(fig)
file = '/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/uncompressed/scenario/formatted_json/tfrecord-00002-of-01000_5.json'
sim = Simulation(file, start_time = 0)
scenario = sim.getScenario()
for i in range(90):
    img = np.array(scenario.getImage(None, render_goals=True), copy=False)
    plt.imshow(img)
    cam.snap()
    sim.waymo_step()

animation = cam.animate(interval=50)
animation.save(f'{os.basename(file)}.mp4')