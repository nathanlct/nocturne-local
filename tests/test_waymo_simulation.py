import os

import matplotlib.pyplot as plt
import numpy as np
from nocturne import Simulation

import time

os.environ["DISPLAY"] = ":0.0"
# sim = Simulation(scenarioFilePath='./nocturne_utils/output.json', startTime=10, showPeds=True, showCyclists=True)
sim = Simulation(scenarioPath='./nocturne_utils/output.json',)
scenario = sim.getScenario()
import ipdb; ipdb.set_trace()

sim.step(0.1)
# while True:
#     sim.step(0.01)
#     sim.render()
print('all veh IDS are', [veh.getID() for veh in scenario.getVehicles()])
# print('all ped IDS are', [veh.getID() for veh in scenario.getPedestrians()])
# print('all cyclist IDS are', [veh.getID() for veh in scenario.getCyclists()])
expert_action = scenario.getExpertAction(scenario.getVehicles()[0].getID(), 10)
print('expert action of vehicle 0 is ', expert_action)
print(scenario.getValidExpertStates(scenario.getVehicles()[0].getID()))
print('is the expert action valid or outside the valid states', scenario.hasExpertAction(scenario.getVehicles()[0].getID(), 10))

img = np.array(scenario.getImage(scenario.getVehicles()[0], renderGoals=True), copy=False)
plt.figure()
plt.imshow(img)
plt.savefig('goalImg.png')
print('saved at ./goalImg.png')

img = np.array(scenario.getCone(scenario.getVehicles()[0], 1.58, 0.0), copy=False)
plt.figure()
plt.imshow(img)
plt.savefig('egoImg.png')
print('saved at ./egoImg.png')

# step a vehicle a bunch of times to confirm that stepping works
print('agent position before step', scenario.getVehicles()[0].getPosition().x, scenario.getVehicles()[0].getPosition().y)
scenario.getVehicles()[0].setAccel(1.5)
for i in range(100):
    sim.step(1.0)
    print(scenario.getVehicles()[0].getSpeed())
    # print('accel is ', scenario.getVehicles()[0].getAccel())
print('agent position after step', scenario.getVehicles()[0].getPosition().x, scenario.getVehicles()[0].getPosition().y)  

img = np.array(scenario.getCone(scenario.getVehicles()[0], 1.58, 0.0), copy=False)
plt.figure()
plt.imshow(img)
plt.savefig('egoImgAfterStep.png')
print('saved at ./egoImgAfterStep.png')

# objects = sim.getRoadObjects()
# print(objects)
# print([obj.getWidth() for obj in objects])

# arr = np.array(sim.getConePixels(), copy=False)
# print(arr.shape, np.max(arr), np.min(arr), arr.dtype)

# plt.figure()
# plt.imshow(arr)
# plt.show()