import os

import matplotlib.pyplot as plt
import numpy as np
from nocturne import Simulation

import time

os.environ["DISPLAY"] = ":0.0"
sim = Simulation(scenarioPath='./scenarios/four_car_intersection.json')
scenario = sim.getScenario()

sim.step(0.1)


objects = scenario.getRoadObjects()
print(objects)

cone = np.array(scenario.getCone(scenario.getVehicles()[0], 3.14/2.0, 0.0), copy=False)
plt.figure()
plt.imshow(cone)
plt.savefig('cone.png')

obj = objects[-1]
img = np.array(scenario.getImage(obj, renderGoals=True), copy=False)
plt.figure()
plt.imshow(img)
plt.savefig('goalImg.png')
print('saved at ./goalImg.png')

# now step forward a huge number of times and check that the cone is advanced
print('agent position before step', scenario.getVehicles()[0].getPosition().x(), scenario.getVehicles()[0].getPosition().y())
scenario.getVehicles()[0].setAccel(1.5)
for i in range(100):
    sim.step(1.0)
    print(scenario.getVehicles()[0].getSpeed())
print('agent position after step', scenario.getVehicles()[0].getPosition().x(), scenario.getVehicles()[0].getPosition().y())  

img = np.array(scenario.getCone(scenario.getVehicles()[0], 1.58, 0.0), copy=False)
plt.figure()
plt.imshow(img)
plt.savefig('egoImgAfterStep.png')
print('saved at ./egoImgAfterStep.png')

# now lets test for collisions
# grab a vehicle and place it on top of another vehicle
sim = Simulation(scenarioPath='./scenarios/four_car_intersection.json')
scenario = sim.getScenario()
veh0 = scenario.getVehicles()[0]
veh1 = scenario.getVehicles()[1]
veh2 = scenario.getVehicles()[2]
# TODO(ev this fails unless the shift is non-zero)
veh1.setPosition(veh0.getPosition().x() + 0.001, veh0.getPosition().y())
sim.step(0.000001)
assert veh1.getCollided() == True, 'vehicle1 should have collided after being placed on vehicle 0'
assert veh0.getCollided() == True, 'vehicle0 should have collided after vehicle 0 was placed on it'
assert veh2.getCollided() == False, 'vehicle2 should not have collided'

# confirm that this is still true a time-step later
sim.step(0.000001)
assert veh1.getCollided() == True, 'vehicle1 should have collided after being placed on vehicle 0'
assert veh0.getCollided() == True, 'vehicle0 should have collided after vehicle 0 was placed on it'
assert veh2.getCollided() == False, 'vehicle2 should not have collided'

# now offset them slightly and do the same thing again
sim = Simulation(scenarioPath='./scenarios/four_car_intersection.json')
scenario = sim.getScenario()
veh0 = scenario.getVehicles()[0]
veh1 = scenario.getVehicles()[1]
veh2 = scenario.getVehicles()[2]
veh0 = scenario.getVehicles()[0]
veh1 = scenario.getVehicles()[1]
veh1.setPosition(veh0.getPosition().x() + 0.2, veh0.getPosition().y() + 0.2)
sim.step(0.000001)
assert veh1.getCollided() == True, 'vehicle1 should have collided after being placed on vehicle 0'
assert veh0.getCollided() == True, 'vehicle0 should have collided after vehicle 0 was placed on it'
assert veh2.getCollided() == False, 'vehicle2 should not have collided'

# now offset them more and do the same thing again
sim = Simulation(scenarioPath='./scenarios/four_car_intersection.json')
scenario = sim.getScenario()
veh0 = scenario.getVehicles()[0]
veh1 = scenario.getVehicles()[1]
veh2 = scenario.getVehicles()[2]
veh0 = scenario.getVehicles()[0]
veh1 = scenario.getVehicles()[1]
veh1.setPosition(veh0.getPosition().x() + 10.0, veh0.getPosition().y() + 30.0)
sim.step(0.000001)
cone = np.array(scenario.getCone(veh0, 2 * np.pi, 0.0), copy=False)
plt.figure()
plt.imshow(cone)
plt.savefig('overlapping_veh.png')
assert veh1.getCollided() == False, 'vehicle1 should have collided after being placed on vehicle 0'
assert veh0.getCollided() == False, 'vehicle0 should have collided after vehicle 0 was placed on it'
assert veh2.getCollided() == False, 'vehicle2 should not have collided'
