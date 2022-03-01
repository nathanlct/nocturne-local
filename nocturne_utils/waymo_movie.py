# TODO(ev) make this efficient and called by step rather than re-rendering the scene
import os

from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np

from nocturne import Simulation
os.environ["DISPLAY"] = ":0.0"

fig = plt.figure()
cam = Camera(fig)
path = '/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/uncompressed/scenario/formatted_json'
files = os.listdir(path)
file = os.path.join(path, files[np.random.randint(len(files))])
sim = Simulation(file, start_time = 0)
scenario = sim.getScenario()
for i in range(90):
    img = np.array(scenario.getImage(None, render_goals=True), copy=False)
    plt.imshow(img)
    cam.snap()
    sim.waymo_step()

animation = cam.animate(interval=50)
animation.save(f'{os.path.basename(file)}.mp4')