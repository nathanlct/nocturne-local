from collections import OrderedDict
import numpy as np

from nocturne_utils.nocturne_utils import angle_between

class Subscriber(object):
    # TODO(eugenevinitsky) just pass the simulation
    def __init__(self, cfg, scenario, simulation):
        self.simulation = simulation
        self.scenario = scenario
        self.cfg = cfg
        self.vehicle_subscriber = VehicleObjectSubscriber(cfg.object_subscriber,
                                                  scenario, simulation, use_local_coords=cfg.use_local_coordinates)
        self.ego_subscriber = EgoSubscriber(cfg.ego_subscriber, scenario,
                                            simulation, use_local_coords=cfg.use_local_coordinates)
        self.road_subscriber = RoadObjectSubscriber(cfg.ego_subscriber, scenario,
                                            simulation, use_local_coords=cfg.use_local_coordinates)
        # TODO(eugenevinitsky) this will change when there's a finite number of visible vehicles
        self.max_num_vehicles = len(self.scenario.getVehicles())
        object_cfg = cfg.object_subscriber
        if cfg.use_local_coordinates:
            # speed, obj dist, obj angle, heading
            self.num_vehicle_elem = object_cfg.include_speed + 2 * object_cfg.include_pos + object_cfg.include_heading
        else:
            # speed, obj pos, distance to object, heading
            self.num_vehicle_elem = object_cfg.include_speed + 2 * object_cfg.include_pos + object_cfg.include_heading + 1

    def get_obs(self, object):
        obs_dict = OrderedDict()
        if self.cfg.include_ego_state:
            obs_dict.update(self.ego_subscriber.get_obs(object))
        # TODO(eugenevinitsky) add this method
        if self.cfg.include_visible_object_state:
            # concatenate all the objects together but don't duplicate yourself
            # TODO(eugenevinitsky) you want this to be only visible objects
            # TODO(eugenevinitsky) instead of all objects
            if self.max_num_vehicles - 1 > 0:
                obs_dict['visible_objects'] = np.zeros((self.max_num_vehicles - 1) * self.num_vehicle_elem)
                if len(self.scenario.getVehicles()) > 1:
                    obs_dict_list = [self.vehicle_subscriber.get_obs(veh_obj, object) for veh_obj in self.scenario.getVehicles() if veh_obj.getID() != object.getID()]
                    # sort the list by angle if in local coordinates
                    if self.cfg.use_local_coordinates:
                        obs_dict_list = sorted(obs_dict_list, key=lambda x: x['obj_angle'])
                    visible_obs_feat = np.concatenate([np.hstack(list(veh_obs_dict.values())) for veh_obs_dict in obs_dict_list])
                    obs_dict['visible_objects'][0 : (len(self.scenario.getVehicles()) - 1) * self.num_vehicle_elem] = visible_obs_feat

        if self.cfg.include_road_edges:
            # get all objects that are not vehicles
            # TODO(eugenevinitsky) make a method to just get the roads
            obs_dict_list = [self.road_subscriber.get_obs(road_obj, object) for road_obj in self.scenario.getRoadObjects() if road_obj.getType() == 'Object']
            # sort the list by angle if in local coordinates
            if self.cfg.use_local_coordinates:
                obs_dict_list = sorted(obs_dict_list, key=lambda x: x['obj_angle'])
            if len(obs_dict_list) > 0:
                obs_dict['road_objects'] = np.concatenate([np.hstack(list(road_obs_dict.values())) for road_obs_dict in obs_dict_list])

        return obs_dict


class VehicleObjectSubscriber(object):
    def __init__(self, cfg, scenario, simulation, use_local_coords):
        self.cfg = cfg
        self.scenario = scenario
        self.simulation = simulation
        self.use_local_coords = use_local_coords

    def get_obs(self, object, observing_object):
        """[summary]

        Args:
            object ([type]): [description]
            observing_object (VehicleObj): Object used to construct local coordinates

        Returns:
            [type]: [description]
        """
        obs_dict = OrderedDict()
        if self.use_local_coords:
            # position and angle used to define the local coordinate frame
            heading_rad = observing_object.getHeading()
        observing_pos = observing_object.getPosition()
        observing_pos = np.array([observing_pos.x, observing_pos.y])
        if self.cfg.include_speed:
            obs_dict['curr_speed'] = np.array([object.getSpeed()])
        if self.cfg.include_pos:
            pos = object.getPosition()
            # if we are using local coordinates, it does not make sense
            # to include a global positions
            if not self.use_local_coords:
                obs_dict['pos'] = np.array([pos.x, pos.y])
                # TODO(eugenevinitsky) this is necessary to form costs for Box DDP 
                # TODO(eugenevinitsky) but we might not want it normally
                obs_dict['obj_dist'] = np.linalg.norm(np.array([pos.x, pos.y]) - observing_pos)
            else:
                # use radial coordinates
                obs_dict['obj_dist'] = np.linalg.norm(np.array([pos.x, pos.y]) - observing_pos)
                obs_dict['obj_angle'] = angle_between(np.array([pos.x, pos.y]) - observing_pos,
                                                       np.array([np.cos(heading_rad), np.sin(heading_rad)])) * 180 / np.pi

        if self.cfg.include_heading:
            if self.use_local_coords:
                obs_dict['heading'] = np.array([(heading_rad - object.getHeading()) * 180 / np.pi]) % 360
            else:
                obs_dict['heading'] = np.array([object.getHeading() * 180 / np.pi]) % 360
        return obs_dict


class RoadObjectSubscriber(object):
    def __init__(self, cfg, scenario, simulation, use_local_coords):
        self.cfg = cfg
        self.scenario = scenario
        self.simulation = simulation
        self.use_local_coords = use_local_coords

    def get_obs(self, object, observing_object):
        """[summary]

        Args:
            object ([type]): [description]
            observing_object (VehicleObj): Object used to construct local coordinates

        Returns:
            [type]: [description]
        """
        small_angle = 0.0002
        obs_dict = OrderedDict()
        # if self.use_local_coords:

        # position of observing vehicle used to define the local coordinate frame
        observing_heading = observing_object.getHeading()
        observing_pos = observing_object.getPosition()
        observing_pos = np.array([observing_pos.x, observing_pos.y])

        # object values
        pos = object.getPosition()
        length = object.getLength()
        heading = object.getHeading()
        # if we are using local coordinates, it does not make sense
        # to include a global positions
        obj_pos = np.array([pos.x, pos.y])
        if not self.use_local_coords:
            obs_dict['pos'] = obj_pos
            obs_dict['heading'] = np.array([heading * 180 / np.pi])
            obs_dict['length'] = np.array([length])
        else:
            # use radial coordinates
            obs_dict['obj_center_dist'] = np.linalg.norm(observing_pos - obj_pos)
            obs_dict['obj_angle'] = angle_between(obj_pos - observing_pos, 
                                                  np.array([np.cos(observing_heading), np.sin(observing_heading)])) * 180 / np.pi
            obs_dict['obj_length'] = np.array([length])
            # this is mod 180 because we don't care which way the edge is pointing
            # we add the small angle to force floating point error to wrap around 180
            obs_dict['obj_rel_angle'] = (np.array([observing_heading - heading]) * 180 / np.pi + small_angle) % 180
            
            # compute the distance to the closest point in the object for the observing object
            # conveniently these objects are all lines so we can just compute the shortest distance
            # to a hyperplane w * x + b = 0 once we construct the normal
            # rotate the heading by 90 degrees to get the normal

        endpoint_1 = np.array([obj_pos[0] + length * np.cos(heading) / 2,
                                    obj_pos[1] + length * np.sin(heading) / 2])
        endpoint_2 = np.array([obj_pos[0] - length * np.cos(heading) / 2,
                                    obj_pos[1] - length * np.sin(heading) / 2])
        # find the closest point on the line and project
        t = max(0, min(1, np.dot(observing_pos - endpoint_1, endpoint_2 - endpoint_1) / (length **2)))
        projection = endpoint_1 + t * (endpoint_2 - endpoint_1)
        new_dist = np.linalg.norm(projection - observing_pos)
        obs_dict['closest_dist'] = new_dist

        return obs_dict


class EgoSubscriber(object):
    def __init__(self, cfg, scenario, simulation, use_local_coords):
        self.cfg = cfg
        self.scenario = scenario
        self.simulation = simulation
        self.use_local_coords = use_local_coords

    def get_obs(self, object):
        obs_dict = OrderedDict()
        heading_deg = object.getHeading()
        ego_pos = object.getPosition()
        if self.cfg.img_view:
            # TODO(eugenevinitsky) include head tilt instead 0.0
            obs_dict['ego_img'] = np.array(self.scenario.getCone(
                object, self.cfg.view_angle, 0.0),
                                             copy=False)
        if self.cfg.include_speed:
            obs_dict['ego_speed'] = np.array([object.getSpeed()])
        # TODO(eugenevinitsky) currently removed these since they aren't necessary when the road objects are given
        # TODO(eugenevinitsky) this is an abstraction break between the road objects subscriber and this
        # since this only makes sense if the road objects are given
        if self.cfg.include_heading and not self.use_local_coords:
            obs_dict['heading'] = np.array([heading_deg * 180 / np.pi]) 
        if self.cfg.include_pos and not self.use_local_coords:
            obs_dict['ego_pos'] = np.array([ego_pos.x, ego_pos.y])
        if self.cfg.include_goal_pos:
            pos = object.getGoalPosition()
            if self.use_local_coords:
                obs_dict['goal_dist'] = np.linalg.norm(np.array([ego_pos.x, ego_pos.y]) - np.array([pos.x, pos.y]))
                obs_dict['goal_angle'] = angle_between(np.array([ego_pos.x, ego_pos.y]) - np.array([pos.x, pos.y]), 
                                                       np.array([np.cos(heading_deg), np.sin(heading_deg)])) * 180 / np.pi
            else:
                obs_dict['goal_pos'] = np.array([pos.x, pos.y])
                obs_dict['goal_dist'] = np.linalg.norm(np.array([ego_pos.x, ego_pos.y]) - np.array([pos.x, pos.y]))
        if self.cfg.include_goal_img:
            obs_dict['goal_img'] = np.array(self.scenario.getImage(object=object, renderGoals=True), copy=False)
        # if self.cfg.include_lane_pos:
        #     obs_dict['lane_pos'] = self.simulation.getLanePos(object)
        return obs_dict

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)
