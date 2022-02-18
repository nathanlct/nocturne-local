import os

import matplotlib.pyplot as plt
import numpy as np
from nocturne import Simulation

import time

os.environ["DISPLAY"] = ":0.0"
# sim = Simulation(scenarioFilePath='./nocturne_utils/output.json', startTime=10, showPeds=True, showCyclists=True)
sim = Simulation(scenarioPath='./nocturne_utils/output.json',)
scenario = sim.getScenario()

print('all veh IDS are', [veh.getID() for veh in scenario.getVehicles()])
# print('all ped IDS are', [veh.getID() for veh in scenario.getPedestrians()])
# print('all cyclist IDS are', [veh.getID() for veh in scenario.getCyclists()])
expert_action = scenario.getExpertAction(scenario.getVehicles()[0].getID(), 10)
print('expert action of vehicle 0 is ', expert_action)
print(scenario.getValidExpertStates(scenario.getVehicles()[0].getID()))
print('is the expert action valid or outside the valid states', scenario.hasExpertAction(scenario.getVehicles()[0].getID(), 10))

import ipdb; ipdb.set_trace()
img = np.array(scenario.getImage(scenario.getVehicles()[3], renderGoals=True), copy=False)
plt.figure()
plt.imshow(img)
plt.savefig('goalImg.png')
print('saved at ./goalImg.png')

img = np.array(scenario.getCone(scenario.getVehicles()[3], 1.58, 0.0), copy=False)
plt.figure()
plt.imshow(img)
plt.savefig('egoImg.png')
print('saved at ./egoImg.png')

img = np.array(scenario.getCone(scenario.getVehicles()[3], 2 * np.pi, 0.0), copy=False)
plt.figure()
plt.imshow(img)
plt.savefig('fullEgoImg.png')
print('saved at ./fullEgoImg.png')

img = np.array(scenario.getCone(scenario.getVehicles()[3], 2 * np.pi, 0.0, False), copy=False)
plt.figure()
plt.imshow(img)
plt.savefig('fullEgoImgNoObscure.png')

# step a vehicle a bunch of times to confirm that stepping works
print('agent position before step', scenario.getVehicles()[3].getPosition().x, scenario.getVehicles()[3].getPosition().y)
scenario.getVehicles()[0].setAccel(1.5)
for i in range(100):
    sim.step(0.2)
    # print('accel is ', scenario.getVehicles()[0].getAccel())
print('agent position after step', scenario.getVehicles()[3].getPosition().x, scenario.getVehicles()[3].getPosition().y)  

img = np.array(scenario.getCone(scenario.getVehicles()[3], 1.58, 0.0), copy=False)
plt.figure()
plt.imshow(img)
plt.savefig('egoImgAfterStep.png')
print('saved at ./egoImgAfterStep.png')

################################
# Collision checking
################################