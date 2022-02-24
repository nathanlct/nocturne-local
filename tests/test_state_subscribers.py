import numpy as np
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display
from nocturne import Simulation
disp = Display()
disp.start()
sim = Simulation(scenario_path='/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/uncompressed/scenario/formatted_json/tfrecord-00002-of-01000_5.json')
scenario = sim.getScenario()
vehs = scenario.getVehicles()
state = scenario.getEgoState(vehs[0])
angle = np.pi/4
state = scenario.getVisibleObjectsState(vehs[0], 2 * np.pi)
img = np.array(scenario.getCone(scenario.getVehicles()[0], 2 * np.pi, 0.0, False), copy=False)
plt.figure()
plt.imshow(img)
plt.savefig('egoImg.png')
print('saved at ./egoImg.png')
print(state)