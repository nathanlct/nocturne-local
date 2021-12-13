"""Each agent receives its observation, a goal position, and tries to get there without colliding."""

from copy import copy
from collections import defaultdict

from gym.spaces import Box
import numpy as np

from nocturne import Simulation
from traitlets.traitlets import default
from nocturne_utils.subscribers import Subscriber
from nocturne_utils.nocturne_utils import angle_between


class BaseEnv(object):
    def __init__(self, cfg, should_terminate=True, rank=0):
        self.simulation = Simulation(cfg.scenario_path)
        self.scenario = self.simulation.getScenario()
        self.vehicles = self.scenario.getVehicles()
        self.subscriber = Subscriber(cfg.subscriber, self.scenario, self.simulation)
        self.cfg = cfg
        self.episode_length = cfg.episode_length
        self.t = 0
        self.step_num = 0
        self.rank=rank
        # If true, agents observations stop being sent back once they're dead
        # If false, a vector of zeros will persist till the episode ends
        self.should_terminate = should_terminate
        # TODO(eugenevinitsky) remove this once the PPO code doesn't have this restriction
        # track dead agents for PPO.
        self.all_vehicle_ids = [veh.getID() for veh in self.vehicles]
        self.dead_feat =  {key: -np.ones_like(value) for key, value in self.subscriber.get_obs(self.vehicles[0]).items()}

    @property
    def observation_space(self):
        pass

    @property
    def action_space(self):
        pass

    def apply_actions(self, action_dict):
        for veh_obj in self.scenario.getVehicles():
            veh_id = veh_obj.getID()
            if veh_id in action_dict.keys():
                action = action_dict[veh_id]
                if 'accel' in action.keys():
                    veh_obj.setAccel(action['accel'])
                if 'turn' in action.keys():
                    veh_obj.setSteeringAngle(action['turn'])
                # if 'tilt_view' in action.keys():
                #     self.simulation.applyViewTilt(action['tilt_view'])

    def step(self, action_dict):
        obs_dict = {}
        rew_dict = {}
        done_dict = {}
        info_dict = defaultdict(dict)
        rew_cfg = self.cfg.rew_cfg
        self.apply_actions(action_dict)
        self.simulation.step(self.cfg.dt)
        self.t += self.cfg.dt
        self.step_num += 1
        objs_to_remove = []
        for veh_obj in self.simulation.getScenario().getVehicles():
            veh_id = veh_obj.getID()
            obs_dict[veh_id] = self.subscriber.get_obs(veh_obj)
            rew_dict[veh_id] = 0
            done_dict[veh_id] = False
            info_dict[veh_id]['goal_achieved'] = False
            info_dict[veh_id]['collided'] = False
            obj_pos = veh_obj.getPosition()
            obj_pos = np.array([obj_pos.x, obj_pos.y])
            goal_pos = veh_obj.getGoalPosition()
            goal_pos = np.array([goal_pos.x, goal_pos.y])
            ######################## Compute rewards #######################
            # TODO(eugenevinitsky) this is never achieved because this is in meters but the goal tolerance is in pixels or something pixel-y
            if np.linalg.norm(goal_pos - obj_pos) < rew_cfg.goal_tolerance:
                rew_dict[veh_id] += np.abs(rew_cfg.goal_achieved_bonus)
                # TODO(eugenevinitsky) temporarily disabled the done condition as this crushes exploration around a goal
                # done_dict[veh_id] = True
                info_dict[veh_id]['goal_achieved'] = True
            if rew_cfg.shaped_goal_distance:
                # the minus one is to ensure that it's not beneficial to collide
                rew_dict[veh_id] -= ((np.linalg.norm((goal_pos - obj_pos) / (400 * 1.5), ord=2) ** 2) - 1)
            if rew_cfg.speed_rew:
                rew_dict[veh_id] += veh_obj.getSpeed() * rew_cfg.speed_rew_coef
            # now if we're using waypoints, compute a waypoint reward and add the waypoint into the obs
            # we don't need to add this penalty if we've actually made it to the goal though
            if rew_cfg.dense_waypoints:
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
            ######################## Handle potential done conditions #######################
            # achieved our goal
            if info_dict[veh_id]['goal_achieved']:
                done_dict[veh_id] = True
            # we have gone off-screen or off road!
            if not self.scenario.isVehicleOnRoad(veh_obj):
                done_dict[veh_id] = True
            if veh_obj.getCollided():
                info_dict[veh_id]['collided'] = True
                rew_dict[veh_id] -= np.abs(rew_cfg.collision_penalty)
                done_dict[veh_id] = True
            # remove the vehicle so that its trajectory doesn't continue. This is important
            # in the multi-agent setting.
            if done_dict[veh_id]:
                objs_to_remove.append(veh_obj)
        
        for veh_obj in objs_to_remove:
            self.scenario.removeObject(veh_obj)
        
        # TODO(eugenevinitsky) remove this once the PPO code doesn't have this restriction
        # track dead agents for PPO.
        if not self.should_terminate:
            for veh_id in self.all_vehicle_ids:
                if veh_id in obs_dict.keys():
                    continue
                else:
                    obs_dict[veh_id] = copy(self.dead_feat)
                    rew_dict[veh_id] = 0
                    done_dict[veh_id] = True
                    info_dict[veh_id] = {'goal_achieved': False}

        if self.cfg.rew_cfg.shared_reward:
            total_reward = np.sum([rew_dict[key] for key in rew_dict.keys()])
            rew_dict = {key: total_reward for key in rew_dict.keys()}
        
        if self.step_num >= self.episode_length:
            done_dict = {key: True for key in done_dict.keys()}

        all_done = True
        for value in done_dict.values():
            all_done *= value
        done_dict['__all__'] = all_done

        return obs_dict, rew_dict, done_dict, info_dict

    def reset(self):
        self.t = 0
        self.step_num = 0
        # TODO(eugenevinitsky) remove this once the PPO code doesn't have this restriction
        # track dead agents for PPO.
        self.simulation.reset()
        self.scenario = self.simulation.getScenario()
        self.vehicles = self.scenario.getVehicles()
        self.all_vehicle_ids = [veh.getID() for veh in self.vehicles]
        self.subscriber = Subscriber(self.cfg.subscriber, self.scenario, self.simulation)
        # initialize the vehicle speeds
        # TODO(eugenevinitsky) make this more general
        for veh_obj in self.simulation.getScenario().getVehicles():
            veh_id = veh_obj.getID()
            veh_obj.setSpeed(self.cfg.initial_speed)
            if self.cfg.randomize_goals:
                invalid_goal = True
                obj_pos = veh_obj.getPosition()
                obj_pos = np.array([obj_pos.x, obj_pos.y])
                new_goal = None
                while invalid_goal:
                    new_goal = 800 * (np.random.uniform(size=(2,)) - 0.5)
                    on_road = self.scenario.isPointOnRoad(new_goal[0], new_goal[1]) 
                    far_away = np.linalg.norm(obj_pos - new_goal) > 200
                    # TODO(eugenevinitsky) make this more general
                    in_intersection = (np.abs(new_goal[0]) < 40) and (np.abs(new_goal[1] < 40))
                    if on_road and far_away and not in_intersection:
                        max_index = np.argmax(np.abs(new_goal))
                        # move it to near the end of the road
                        new_goal[max_index] = 380 * np.sign(new_goal[max_index])
                        invalid_goal = False

                veh_obj.setGoalPosition(new_goal[0], new_goal[1])

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

        obs_dict = {}
        for veh_obj in self.simulation.getScenario().getVehicles():
            veh_id = veh_obj.getID()
            obs_dict[veh_id] = self.subscriber.get_obs(veh_obj)

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
                self.dead_feat['waypoint_angle'] = -np.ones_like(waypoint_angle)
                self.dead_feat['waypoint_dist'] = -np.ones_like(obs_dict[veh_id]['waypoint_dist'])
        
        return obs_dict
    def render(self, mode=None):
        # TODO(eugenevinitsky) this should eventually return a global image instead of this hack
        return np.array(self.scenario.getImage(object=None, renderGoals=True), copy=False)

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

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
