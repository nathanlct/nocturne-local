import os

import matplotlib.pyplot as plt
import numpy as np
from nocturne import Simulation

# os.environ["DISPLAY"] = ":0.0"

sim = Simulation(render=False, scenarioPath='./scenarios/basic.json')

objects = sim.getRoadObjects()
print(objects)
print([obj.getWidth() for obj in objects])

arr = np.array(sim.getConePixels(), copy=False)
print(arr.shape, np.max(arr), np.min(arr), arr.dtype)

plt.figure()
plt.imshow(arr)
plt.savefig('example.png')
