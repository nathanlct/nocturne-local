import numpy as np
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display
from nocturne import Simulation
import time

import os


def test_state_subscribers():

    os.environ["DISPLAY"] = ":0.0"
    sim = Simulation(scenario_path='tests/scenario_test.json')
    scenario = sim.getScenario()
    vehs = scenario.getVehicles()

    ##################################
    # Test ego state getter
    #################################
    state = scenario.egoStateObservation(vehs[0])
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
    max_num_visible_objects = scenario.getMaxNumVisibleKineticObjects()
    num_object_states = scenario.getKineticObjectFeatureSize()
    max_num_visible_road_points = scenario.getMaxNumVisibleRoadPoints()
    num_road_point_states = scenario.getRoadPointFeatureSize()
    max_num_visible_stop_signs = scenario.getMaxNumVisibleStopSigns()
    num_stop_sign_states = scenario.getStopSignsFeatureSize()
    max_num_visible_tl_signs = scenario.getMaxNumVisibleTrafficLights()
    num_tl_states = scenario.getTrafficLightFeatureSize()
    new_state = scenario.observation(vehs[0], 120.0, 1.99 * np.pi)
    # check that the observed vehicle has the right state
    # the vehicle is 10 meters away northwards, pointed east, we are pointed north
    # so our result should be
    # [1 #valid, 10 #distance, 0 # azimuth, 2 # length, 1 # width, -pi/2 # relative-heading, 5 # relative x-speed
    # -5 * np.sqrt(2) relative y speed]
    # the object is a vehicle so it gets one-hot to [0, 1, 0, 0, 0, 0, 0, 0]
    np.testing.assert_allclose(new_state[0:num_object_states], [
        1, 10.0, 0.0, vehs[1].getLength(), vehs[1].getWidth(), -np.pi / 2, 5.0,
        -5 * np.sqrt(2), 0, 1, 0, 0, 0, 0, 0, 0
    ],
                               rtol=1e-5,
                               atol=1e-5)

    # check that the observed road points are fine they are at [(10, 10), (11, 11)]
    # and are "road edge = 3, road edge = 3"
    # since these are road edges the one hot encoding will be [0, 0, 0, 1, 0, 0, 0]
    road_point_state = new_state[max_num_visible_objects *
                                 num_object_states:max_num_visible_objects *
                                 num_object_states + num_road_point_states * 2]
    np.testing.assert_allclose(road_point_state, [
        1, 10.0 * np.sqrt(2), -np.pi / 4, 0, 0, 0, 1, 0, 0, 0, 1,
        11.0 * np.sqrt(2), -np.pi / 4, 0, 0, 0, 1, 0, 0, 0
    ],
                               rtol=1e-5,
                               atol=1e-5)

    # now do the same thing with the stop sign at (8, 8)
    new_start_point = max_num_visible_objects * num_object_states + max_num_visible_road_points * num_road_point_states + max_num_visible_tl_signs * num_tl_states
    stop_sign_state = new_state[new_start_point:new_start_point +
                                num_stop_sign_states]
    np.testing.assert_allclose(stop_sign_state,
                               [1, 8.0 * np.sqrt(2), -np.pi / 4],
                               rtol=1e-5,
                               atol=1e-5)

    ########################## now do the sames but with a partially obscured view, we should only see the vehicle and nothing else
    new_state = scenario.observation(vehs[0], 120, 0.1)
    # vehicle
    np.testing.assert_allclose(new_state[0:num_object_states], [
        1, 10.0, 0.0, vehs[1].getLength(), vehs[1].getWidth(), -np.pi / 2, 5.0,
        -5 * np.sqrt(2), 0, 1, 0, 0, 0, 0, 0, 0
    ],
                               rtol=1e-5,
                               atol=1e-5)
    # road point, it shouldn't be visible
    road_point_state = new_state[max_num_visible_objects *
                                 num_object_states:max_num_visible_objects *
                                 num_object_states + num_road_point_states * 2]
    np.testing.assert_allclose(road_point_state,
                               [0] * 2 * num_road_point_states,
                               rtol=1e-5,
                               atol=1e-5)

    ########################## now rotate the vehicle so it sees the road points but not the vehicle ##############
    vehs[0].setHeading(np.pi / 4)
    new_state = scenario.observation(vehs[0], 120, 0.1)
    # vehicle
    np.testing.assert_allclose(new_state[0:num_object_states],
                               [0] * num_object_states,
                               rtol=1e-5,
                               atol=1e-5)
    # check that the observed road points are fine they are at [(10, 10), (11, 11)]
    # and are "road edge = 3, road edge = 3"
    road_point_state = new_state[max_num_visible_objects *
                                 num_object_states:max_num_visible_objects *
                                 num_object_states + num_road_point_states * 2]
    np.testing.assert_allclose(road_point_state, [
        1, 10.0 * np.sqrt(2), 0, 0, 0, 0, 1, 0, 0, 0, 1, 11.0 * np.sqrt(2), 0,
        0, 0, 0, 1, 0, 0, 0
    ],
                               rtol=1e-5,
                               atol=1e-5)


if __name__ == '__main__':
    test_state_subscribers()
