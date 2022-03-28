import numpy as np
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display
from nocturne import Simulation
import time

disp = Display()
disp.start()
sim = Simulation(scenario_path='./scenario_test.json')
scenario = sim.getScenario()
vehs = scenario.getVehicles()

##################################
# Test ego state getter
#################################
state = scenario.getEgoState(vehs[0])
# speed, goal dist, goal angle, length, width
np.testing.assert_allclose(state, [
    5.0 * np.sqrt(2), 100 * np.sqrt(2), 3 * np.pi / 4, vehs[0].getLength(),
    vehs[0].getWidth()
],
                           rtol=1e-5)
np.testing.assert_allclose(vehs[0].getHeading(), np.pi / 2)

##################################
# Test general state getter when we see every object
#################################
max_num_visible_objects = scenario.getMaxNumVisibleObjects()
num_object_states = 6
max_num_visible_road_points = scenario.getMaxNumVisibleRoadPoints()
num_road_point_states = 3
max_num_visible_stop_signs = scenario.getMaxNumVisibleStopSigns()
num_stop_sign_states = 2
max_num_visible_tl_signs = scenario.getMaxNumVisibleTLSigns()
num_tl_states = 3
new_state = scenario.getVisibleState(vehs[0], 2 * np.pi)
# check that the observed vehicle has the right state
# the vehicle is 10 meters away northwards, pointed east, we are pointed north
np.testing.assert_allclose(
    new_state[0:num_object_states],
    [10.0, 0.0, -np.pi / 2, 5.0, vehs[1].getLength(), vehs[1].getWidth()],
    rtol=1e-5,
    atol=1e-5)

# check that the observed road points are fine they are at [(10, 10), (11, 11)]
# and are "road edge = 3, road edge = 3"
road_point_state = new_state[max_num_visible_objects *
                             num_object_states:max_num_visible_objects *
                             num_object_states + num_road_point_states * 2]
np.testing.assert_allclose(
    road_point_state,
    [10.0 * np.sqrt(2), -np.pi / 4, 3, 11.0 * np.sqrt(2), -np.pi / 4, 3],
    rtol=1e-5,
    atol=1e-5)

# now do the same thing with the stop sign at (8, 8)
new_start_point = max_num_visible_objects * num_object_states + max_num_visible_road_points * num_road_point_states
stop_sign_state = new_state[new_start_point:new_start_point +
                            num_stop_sign_states]
np.testing.assert_allclose(stop_sign_state, [8.0 * np.sqrt(2), -np.pi / 4],
                           rtol=1e-5,
                           atol=1e-5)

########################## now do the same but with a partially obscured view, we should only see the vehicle and nothing else
new_state = scenario.getVisibleState(vehs[0], 0.1)
# vehicle
np.testing.assert_allclose(
    new_state[0:num_object_states],
    [10.0, 0.0, -np.pi / 2, 5.0, vehs[1].getLength(), vehs[1].getWidth()],
    rtol=1e-5,
    atol=1e-5)
# road point, it shouldn't be visible
road_point_state = new_state[max_num_visible_objects *
                             num_object_states:max_num_visible_objects *
                             num_object_states + num_road_point_states * 2]
np.testing.assert_allclose(road_point_state, [-100] * 6, rtol=1e-5, atol=1e-5)

########################## now rotate the vehicle so it sees the road points but not the vehicle ##############
vehs[0].setHeading(np.pi / 4)
new_state = scenario.getVisibleState(vehs[0], 0.1)
print(vehs[0].getHeading())
# vehicle
np.testing.assert_allclose(new_state[0:num_object_states], [-100] * 6,
                           rtol=1e-5,
                           atol=1e-5)
# check that the observed road points are fine they are at [(10, 10), (11, 11)]
# and are "road edge = 3, road edge = 3"
road_point_state = new_state[max_num_visible_objects *
                             num_object_states:max_num_visible_objects *
                             num_object_states + num_road_point_states * 2]
np.testing.assert_allclose(road_point_state,
                           [10.0 * np.sqrt(2), 0, 3, 11.0 * np.sqrt(2), 0, 3],
                           rtol=1e-5,
                           atol=1e-5)

img = np.array(scenario.getCone(scenario.getVehicles()[0], 2 * np.pi, 0.0,
                                False),
               copy=False)

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

sim.reset()
times = []
for _ in range(5):
    t = time.time()
    sim.waymo_step()
    diff = time.time() - t
    times.append(diff)

print('run time for waymo step is ', np.mean(times))