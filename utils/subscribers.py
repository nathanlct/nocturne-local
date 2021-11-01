from collections import OrderedDict
import numpy as np

class Subscriber(object):
    # TODO(eugenevinitsky) just pass the simulation
    def __init__(self, cfg, scenario, simulation):
        self.simulation = simulation
        self.scenario = scenario
        self.cfg = cfg
        self.object_subscriber = ObjectSubscriber(cfg.object_subscriber, scenario, simulation)
        self.ego_subscriber = EgoSubscriber(cfg.ego_subscriber, scenario, simulation)

    def get_obs(self, object):
        obs_dict = OrderedDict()
        if self.cfg.include_ego_state:
            obs_dict.update(self.ego_subscriber.get_obs(object))
        # TODO(eugenevinitsky) add this method
        if self.cfg.include_visible_object_state:
            obs_dict['visible_objects'] = OrderedDict({
                obj_id: self.object_subscriber.get_obs(obj)
                for obj_id, obj in self.scenario.getVisibleObjects(object)
            })
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
                object, self.cfg.view_angle, 0.0), copy=False)
        if self.cfg.include_speed:
            obs_dict['ego_speed'] = np.array([object.getSpeed()])
        if self.cfg.include_pos:
            pos = object.getPosition()
            obs_dict['ego_pos'] = np.array([pos.x, pos.y])
        if self.cfg.include_goal_pos:
            pos = object.getGoalPosition()
            obs_dict['goal_pos'] = np.array([pos.x, pos.y])
        if self.cfg.include_goal_img:
            obs_dict['goal_img'] = np.array(self.scenario.getGoalImage(object), copy=False)
        # if self.cfg.include_lane_pos:
        #     obs_dict['lane_pos'] = self.simulation.getLanePos(object)
        return obs_dict
