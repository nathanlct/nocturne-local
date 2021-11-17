from collections import OrderedDict
import numpy as np


class Subscriber(object):
    # TODO(eugenevinitsky) just pass the simulation
    def __init__(self, cfg, scenario, simulation):
        self.simulation = simulation
        self.scenario = scenario
        self.cfg = cfg
        self.object_subscriber = ObjectSubscriber(cfg.object_subscriber,
                                                  scenario, simulation)
        self.ego_subscriber = EgoSubscriber(cfg.ego_subscriber, scenario,
                                            simulation)
        # TODO(eugenevinitsky) this will change when there's a finite number of visible vehicles
        self.max_num_vehicles = len(self.scenario.getVehicles())
        object_cfg = cfg.object_subscriber
        self.num_vehicle_elem = object_cfg.include_speed + 2 * object_cfg.include_pos + object_cfg.include_heading

    def get_obs(self, object):
        obs_dict = OrderedDict()
        if self.cfg.include_ego_state:
            obs_dict.update(self.ego_subscriber.get_obs(object))
        # TODO(eugenevinitsky) add this method
        if self.cfg.include_visible_object_state:
            # concatenate all the objects together but don't duplicate yourself
            # TODO(eugenevinitsky) you want this to be only visible objects
            # TODO(eugenevinitsky) instead of all objects
            obs_dict['visible_objects'] = np.zeros(self.max_num_vehicles * self.num_vehicle_elem)
            if len(self.scenario.getVehicles()) > 1:
                visible_obs_feat = np.concatenate([np.hstack(list(self.object_subscriber.get_obs(obj).values())) for obj in 
                        self.scenario.getVehicles() if obj.getID() != object.getID()])
                obs_dict['visible_objects'][0 : (len(self.scenario.getVehicles()) - 1) * self.num_vehicle_elem] = visible_obs_feat
        
        return obs_dict


class ObjectSubscriber(object):
    def __init__(self, cfg, scenario, simulation):
        self.cfg = cfg
        self.scenario = scenario
        self.simulation = simulation

    def get_obs(self, object):
        obs_dict = OrderedDict()
        if self.cfg.include_speed:
            obs_dict['curr_speed'] = np.array([object.getSpeed()])
        if self.cfg.include_pos:
            pos = object.getPosition()
            obs_dict['pos'] = np.array([pos.x, pos.y])
        if self.cfg.include_heading:
            obs_dict['heading'] = np.array([object.getHeading()*180/3.14])
        return obs_dict


class EgoSubscriber(object):
    def __init__(self, cfg, scenario, simulation):
        self.cfg = cfg
        self.scenario = scenario
        self.simulation = simulation

    def get_obs(self, object):
        obs_dict = OrderedDict()
        if self.cfg.img_view:
            # TODO(eugenevinitsky) include head tilt instead 0.0
            obs_dict['ego_image'] = np.array(self.scenario.getCone(
                object, self.cfg.view_angle, 0.0),
                                             copy=False)
        if self.cfg.include_speed:
            obs_dict['ego_speed'] = np.array([object.getSpeed()])
        if self.cfg.include_heading:
            obs_dict['heading'] = np.array([object.getHeading()*180/3.14])
        if self.cfg.include_pos:
            pos = object.getPosition()
            obs_dict['ego_pos'] = np.array([pos.x, pos.y])
        if self.cfg.include_goal_pos:
            pos = object.getGoalPosition()
            obs_dict['goal_pos'] = np.array([pos.x, pos.y])
        if self.cfg.include_goal_img:
            obs_dict['goal_img'] = np.array(self.scenario.getGoalImage(object),
                                            copy=False)
        # if self.cfg.include_lane_pos:
        #     obs_dict['lane_pos'] = self.simulation.getLanePos(object)
        return obs_dict
