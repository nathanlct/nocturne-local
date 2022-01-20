from collections import OrderedDict, defaultdict

import numpy as np

from envs.base_env import BaseEnv
from nocturne_utils.nocturne_utils import angle_between

class WaypointEnv(BaseEnv):
    def __init__(self, cfg, should_terminate=True, rank=0):
        """[summary]

        Args:
            cfg ([type]): [description]
            should_terminate (bool, optional): see super-class
            rank (int, optional): [description]. Defaults to 0.
        """
        super().__init__(cfg, should_terminate=should_terminate, rank=rank)
        num_waypoints = self.cfg.rew_cfg.num_visible_waypoints
        self.dead_feat['waypoint_angle'] = -np.ones(num_waypoints)
        self.dead_feat['waypoint_dist'] = -np.ones(num_waypoints)

    def step(self, action):
        # we need to compute our rewards and observations before calling super, because the base environment
        # will remove objects that have crashed or completed their goalsc
        rew_dict = defaultdict(int)
        obs_dict = defaultdict(OrderedDict)
        rew_cfg = self.cfg.rew_cfg
        for veh_obj in self.simulation.getScenario().getVehicles():
            ####################################################
            # Compute the waypoint reward
            ####################################################
            veh_id = veh_obj.getID()
            obj_pos = veh_obj.getPosition()
            obj_pos = np.array([obj_pos.x, obj_pos.y])
            waypoints, waypoint_idx = self.waypoints_dict[veh_id]
            # compute a small penalty for distance from the waypoint
            curr_waypoint = waypoints[waypoint_idx]
            curr_dist = np.linalg.norm(obj_pos - curr_waypoint)
            # we are close enough to the waypoint 
            if curr_dist < rew_cfg.goal_tolerance:
                # this needs to be smaller than the goal reward so that the agent prefers getting
                # to the goal and is willing to ignore waypoints
                rew_dict[veh_id] += 0.2 # TODO(eugenevinitsky) remove hardcoding
                self.waypoints_dict[veh_id][1] += 1
                self.waypoints_dict[veh_id][1] = min(self.waypoints_dict[veh_id][1], self.waypoints_dict[veh_id][0].shape[0] - 1)
                curr_waypoint = waypoints[waypoint_idx]
                curr_dist = np.linalg.norm(obj_pos - curr_waypoint)
            rew_dict[veh_id] -= 0.0001 * curr_dist

            ####################################################
            # Construct the waypoint observations
            ####################################################
            num_waypoints = rew_cfg.num_visible_waypoints
            obs_waypoints = waypoints[waypoint_idx: min(waypoint_idx + num_waypoints, waypoints.shape[0])].copy()
            # put in local coords
            obs_waypoints -= obj_pos[np.newaxis, :]
            if obs_waypoints.shape[0] < num_waypoints:
                if obs_waypoints.shape[0] > 0:
                    obs_waypoints = np.vstack((obs_waypoints, -800 * np.ones((num_waypoints - obs_waypoints.shape[0], 2))))
                else:
                    obs_waypoints = -800 * np.ones((num_waypoints, 2))
            # obs_dict[veh_id]['waypoints'] = obs_waypoints.ravel()
            obs_dict[veh_id]['waypoint_dist'] = np.linalg.norm(obs_waypoints, axis=1).ravel()
            heading_deg = veh_obj.getHeading()
            waypoint_angle = np.zeros(num_waypoints)
            for j, waypoint in enumerate(obs_waypoints):
                waypoint_angle[j] = angle_between(waypoint, np.array([np.cos(heading_deg), np.sin(heading_deg)])) * 180 / np.pi
            obs_dict[veh_id]['waypoint_angle'] = waypoint_angle
        
        super_obs, super_rew, done, info = super().step(action)
        # insert the additional waypoint values into super obs
        for key, rew in rew_dict.items():
            super_rew[key] += rew
            super_obs[key].update(obs_dict[key])
        return super_obs, super_rew, done, info

    def reset(self):
        obs_dict = super().reset()
         # okay, now if we want dense waypoints, now is the time to compute them
        if self.cfg.rew_cfg.dense_waypoints:
            self.waypoints_dict = {}
            for veh in self.vehicles:
                initial_pos = np.array([veh.getPosition().x, veh.getPosition().y])
                goal_pos = np.array([veh.getGoalPosition().x, veh.getGoalPosition().y])
                step_size = self.cfg.rew_cfg.dense_waypoint_spacing
                waypoints = self.construct_waypoints(initial_pos, goal_pos, step_size)
                # waypoint position, and a tracker for which waypoint we are on
                waypoint_index = 0
                self.waypoints_dict[veh.getID()] = [waypoints, waypoint_index]

        for veh_obj in self.simulation.getScenario().getVehicles():
            veh_id = veh_obj.getID()
            # TODO(eugenevinitsky) this is bad, this shouldn't be here
            if self.cfg.rew_cfg.dense_waypoints:
                num_waypoints = self.cfg.rew_cfg.num_visible_waypoints
                obs_waypoints = self.waypoints_dict[veh_id][0][0: min(0 + num_waypoints, waypoints.shape[0])].copy()
                # put in local coords
                obj_pos = veh_obj.getPosition()
                obj_pos = np.array([obj_pos.x, obj_pos.y])
                obs_waypoints -= obj_pos[np.newaxis, :]
                if obs_waypoints.shape[0] < num_waypoints:
                    if obs_waypoints.shape[0] > 0:
                        obs_waypoints = np.vstack((obs_waypoints, -800 * np.ones((num_waypoints - obs_waypoints.shape[0], 2))))
                    else:
                        obs_waypoints = -800 * np.ones((num_waypoints, 2))
                obs_dict[veh_id]['waypoint_dist'] = np.linalg.norm(obs_waypoints, axis=1).ravel()
                heading_deg = veh_obj.getHeading()
                waypoint_angle = np.zeros(num_waypoints)
                for j, waypoint in enumerate(obs_waypoints):
                    waypoint_angle[j] = angle_between(waypoint, np.array([np.cos(heading_deg), np.sin(heading_deg)])) * 180 / np.pi
                obs_dict[veh_id]['waypoint_angle'] = waypoint_angle
        
        return obs_dict

    def construct_waypoints(self, initial_pos, final_goal, step_size):
        # TODO(eugenevinitsky) we can make this faster I think
        way_points = []
        curr_distance = np.linalg.norm(initial_pos - final_goal)
        while np.linalg.norm(initial_pos - final_goal) > step_size:
            direction = (np.random.uniform(size=(2,)) - 0.5)
            direction /= np.linalg.norm(direction)
            proposed_pos = initial_pos + direction * step_size
            new_dist = np.linalg.norm(proposed_pos - final_goal)
            # check if the new point is closer and on the road
            if self.scenario.isPointOnRoad(proposed_pos[0], proposed_pos[1]) and new_dist < curr_distance:
                curr_distance = new_dist
                initial_pos = proposed_pos
                way_points.append(initial_pos)
        return np.stack(way_points)
