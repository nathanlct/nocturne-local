import numpy as np
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display
from nocturne import Simulation
import time
disp = Display()
disp.start()
sim = Simulation(scenario_path='/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/uncompressed/scenario/formatted_json/tfrecord-00002-of-01000_5.json')
scenario = sim.getScenario()
vehs = scenario.getVehicles()
state = scenario.getEgoState(vehs[0])
print('ego state is ', state)
angle = np.pi/4
new_state = scenario.getVisibleState(vehs[0], 2 * np.pi)
print('new state is ', new_state)
img = np.array(scenario.getCone(scenario.getVehicles()[0], 2 * np.pi, 0.0, False), copy=False)
plt.figure()
plt.imshow(img)
plt.savefig('egoImg.png')
print('saved at ./egoImg.png')
print(new_state)
times = []
for _ in range(5):
    t = time.time()
    new_state = scenario.getVisibleState(vehs[0], 2 * np.pi)
    diff = time.time() - t
    times.append(diff)

print('run time for full view is is ', np.mean(times))

times = []
for _ in range(5):
    t = time.time()
    new_state = scenario.getVisibleState(vehs[0], 1.58)
    diff = time.time() - t
    times.append(diff)

print('run time for partial view is is ', np.mean(times))

times = []
for _ in range(5):
    t = time.time()
    sim.step(0.1)
    diff = time.time() - t
    times.append(diff)

print('run time for step is ', np.mean(times))


times = []
for _ in range(5):
    t = time.time()
    sim.waymo_step()
    diff = time.time() - t
    times.append(diff)

print('run time for waymo step is ', np.mean(times))


