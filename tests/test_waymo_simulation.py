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
# Vehicle Collision checking
################################
# now lets test for collisions
# grab a vehicle and place it on top of another vehicle
sim = Simulation(scenarioPath='./nocturne_utils/output.json')
scenario = sim.getScenario()
veh0 = scenario.getVehicles()[0]
veh1 = scenario.getVehicles()[1]
veh2 = scenario.getVehicles()[2]
# TODO(ev this fails unless the shift is non-zero)
veh1.setPosition(veh0.getPosition().x + 0.001, veh0.getPosition().y)
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
sim = Simulation(scenarioPath='./nocturne_utils/output.json')
scenario = sim.getScenario()
veh0 = scenario.getVehicles()[0]
veh1 = scenario.getVehicles()[1]
veh2 = scenario.getVehicles()[2]
veh0 = scenario.getVehicles()[0]
veh1 = scenario.getVehicles()[1]
veh1.setPosition(veh0.getPosition().x + 0.2, veh0.getPosition().y + 0.2)
sim.step(0.000001)
assert veh1.getCollided() == True, 'vehicle1 should have collided after being placed overlapping vehicle 0'
assert veh0.getCollided() == True, 'vehicle0 should have collided after vehicle 1 was placed on it'
assert veh2.getCollided() == False, 'vehicle2 should not have collided'

# now offset them more and do the same thing again
sim = Simulation(scenarioPath='./nocturne_utils/output.json')
scenario = sim.getScenario()
veh0 = scenario.getVehicles()[0]
veh1 = scenario.getVehicles()[1]
veh2 = scenario.getVehicles()[2]
veh0 = scenario.getVehicles()[0]
veh1 = scenario.getVehicles()[1]
veh1.setPosition(veh0.getPosition().x + 10.0, veh0.getPosition().y + 30.0)
sim.step(0.000001)
cone = np.array(scenario.getCone(veh0, 2 * np.pi, 0.0), copy=False)
plt.figure()
plt.imshow(cone)
plt.savefig('overlapping_veh.png')
assert veh1.getCollided() == False, 'vehicle1 should not have collided'
assert veh0.getCollided() == False, 'vehicle0 should not have collided'
assert veh2.getCollided() == False, 'vehicle2 should not have collided'

################################
# Road Collision checking
################################
# now check if we place it onto one of the road points that there should be a collision
# find a road edge 
colliding_road_line = None
for roadline in scenario.getRoadLines():
    if roadline.canCollide():
        colliding_road_line = roadline
        break
roadpoints = colliding_road_line.getAllPoints()
start_point = np.array([roadpoints[0].x, roadpoints[0].y])
road_segment_dir = np.array([roadpoints[1].x, roadpoints[1].y]) - np.array([roadpoints[0].x, roadpoints[0].y])
assert np.linalg.norm(road_segment_dir) < 1 # it should be able to fit inside the vehicle
road_segment_angle = np.arctan2(road_segment_dir[1], road_segment_dir[0]) #atan2 is (y, x) not (x,y)
veh0.setHeading(road_segment_angle)

# place the vehicle so that the segment is contained inside of it
new_center = start_point + 0.5 * road_segment_dir
veh0.setPosition(new_center[0], new_center[1])
sim.step(1e-6)
cone = np.array(scenario.getCone(veh0, 2 * np.pi, 0.0), copy=False)
plt.figure()
plt.imshow(cone)
plt.savefig('line_veh_check.png')
assert veh0.getCollided() == True, 'vehicle0 should have collided after a road edge is placed inside it'

# place the vehicle on one of the points so that the road segment intersects with a vehicle edge
sim.reset()
scenario = sim.getScenario()
veh0 = scenario.getVehicles()[0]
veh0.setHeading(road_segment_angle)
veh_length = veh0.getLength()
new_center += veh_length / 2 * road_segment_dir
veh0.setPosition(new_center[0], new_center[1])
sim.step(1e-6)
cone = np.array(scenario.getCone(veh0, 2 * np.pi, 0.0), copy=False)
plt.figure()
plt.imshow(cone)
plt.savefig('line_veh_check2.png')
assert veh0.getCollided() == True, 'vehicle0 should have collided since a road edge intersects it'
