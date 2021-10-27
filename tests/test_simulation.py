import numpy as np
from nocturne import Simulation
import matplotlib.pyplot as plt
import time

sim = Simulation(scenarioPath='./scenarios/intersection.json')
scenario = sim.getScenario()

while True:
    sim.step(0.01)
    sim.render()

objects = scenario.getRoadObjects()
print(objects)

for i, obj in enumerate(objects):
    cone = np.array(scenario.getCone(obj, 3.14/2.0, 0.0), copy=False)
    plt.figure()
    plt.imshow(cone)
    plt.savefig(f'{i}_.png')

# objects = sim.getRoadObjects()
# print(objects)
# print([obj.getWidth() for obj in objects])

# arr = np.array(sim.getConePixels(), copy=False)
# print(arr.shape, np.max(arr), np.min(arr), arr.dtype)

# plt.figure()
# plt.imshow(arr)
# plt.show()