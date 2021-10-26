class Subscriber(object):
    def __init__(self, cfg, simulation):
        self.simulation = simulation
        self.cfg = cfg
        self.object_subscriber(cfg.object_subscriber_cfg, simulation)
        self.ego_subscriber(cfg.ego_subscriber, simulation)

    def get_obs(self, object):
        obs_dict = {}
        if self.cfg.include_ego_state:
            obs_dict.update(self.ego_subscriber.get_obs(object))
        if self.cfg.include_visible_object_state:
            obs_dict['visible_objects'] = {
                obj_id: self.object_subscriber.get_obs(obj)
                for obj_id, obj in self.simulation.getVisibleObjects(object)
            }
        return obs_dict


class ObjectSubscriber(object):
    def __init__(self, cfg, simulation):
        self.cfg = cfg
        self.simulation = simulation

    def get_obs(self, object):
        obs_dict = {}
        if self.cfg.include_speed:
            obs_dict['curr_speed'] = self.simulation.getSpeed(object)
        if self.cfg.include_xy:
            obs_dict['xy'] = self.simulation.getXY(object)
        return obs_dict


class EgoSubscriber(object):
    def __init__(self, cfg, simulation):
        self.cfg = cfg
        self.simulation = simulation

    def get_obs(self, object):
        obs_dict = {}
        if self.cfg.img_view:
            obs_dict['ego_image'] = self.simulation.getCone(
                object, self.cfg.view_angle, object.getHeadTilt())
        if self.cfg.include_speed:
            obs_dict['ego_speed'] = self.simulation.getSpeed(object)
        if self.cfg.include_goal_xy:
            obs_dict['goal_xy'] = self.simulation.getGoalXY(object)
        if self.cfg.include_goal_img:
            obs_dict['goal_img'] = self.simulation.getGoalXY(object)
        if self.cfg.include_lane_pos:
            obs_dict['lane_pos'] = self.simulation.getLanePos(object)
        return obs_dict
