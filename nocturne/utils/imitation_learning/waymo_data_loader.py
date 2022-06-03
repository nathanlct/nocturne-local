"""Dataloader for imitation learning in Nocturne."""
import torch
from pathlib import Path
import numpy as np

from nocturne import Simulation


def _get_waymo_iterator(paths, dataloader_config, scenario_config):
    # if worker has no paths, return an empty iterator
    if len(paths) == 0:
        return

    # load dataloader config
    tmin = dataloader_config.get('tmin', 0)
    tmax = dataloader_config.get('tmax', 90)
    view_dist = dataloader_config.get('view_dist', 80)
    view_angle = dataloader_config.get('view_angle', np.radians(120))
    dt = dataloader_config.get('dt', 0.1)
    expert_action_bounds = dataloader_config.get('expert_action_bounds', [[-3, 3], [-0.7, 0.7]])
    accel_discretization = dataloader_config.get('accel_discretization')
    accel_grid = np.linspace(expert_action_bounds[0][0], expert_action_bounds[0][1], accel_discretization)
    steer_discretization = dataloader_config.get('steer_discretization')
    steer_grid = np.linspace(expert_action_bounds[1][0], expert_action_bounds[1][1], steer_discretization)
    state_normalization = dataloader_config.get('state_normalization', 100)
    n_stacked_states = dataloader_config.get('n_stacked_states', 5)

    while True:
        # select a random scenario path
        scenario_path = np.random.choice(paths)

        # create simulation
        sim = Simulation(str(scenario_path), scenario_config)
        scenario = sim.getScenario()

        # set objects to be expert-controlled
        for obj in scenario.getObjects():
            obj.expert_control = True

        # we are interested in imitating vehicles that moved
        objects_that_moved = scenario.getObjectsThatMoved()
        objects_of_interest = [obj for obj in scenario.getVehicles() if obj in objects_that_moved]

        # initialize values if stacking states
        stacked_state = None
        initial_warmup = n_stacked_states - 1

        # iterate over timesteps and objects of interest
        for time in range(tmin, tmax):
            for obj in objects_of_interest:
                # get state
                ego_state = scenario.ego_state(obj)
                visible_state = scenario.flattened_visible_state(
                    obj, view_dist=view_dist, view_angle=view_angle)
                state = np.concatenate((ego_state, visible_state))

                # normalize state
                state /= state_normalization

                # stack state
                if n_stacked_states > 1:
                    if stacked_state is None:
                        stacked_state = np.zeros(len(state) * n_stacked_states, dtype=state.dtype)
                    stacked_state = np.roll(stacked_state, len(state))
                    stacked_state[:len(state)] = state

                # get expert action
                expert_action = scenario.expert_action(obj, time)
                if expert_action is None:
                    continue
                expert_action = expert_action.numpy()
                # now find the corresponding expert actions in the grids

                # throw out actions containing NaN or out-of-bound values
                if np.isnan(expert_action).any() \
                        or expert_action[0] < expert_action_bounds[0][0] \
                        or expert_action[0] > expert_action_bounds[0][1] \
                        or expert_action[1] < expert_action_bounds[1][0] \
                        or expert_action[1] > expert_action_bounds[1][1]:
                    continue
                # now find it in the grid
                accel = expert_action[0]
                steer = expert_action[1]
                accel_index = find_nearest(accel_grid, accel)
                steer_index = find_nearest(steer_grid, steer)
                # accel_one_hot = np.zeros(accel_discretization)
                # accel_one_hot[accel_index] = 1
                # steer_one_hot = np.zeros(steer_discretization)
                # steer_one_hot[steer_index] = 1

                # yield state and expert action
                if stacked_state is not None:
                    if initial_warmup <= 0:  # warmup to wait for stacked state to be filled up
                        yield (stacked_state, [accel_index, steer_index])
                else:
                    yield (state, [accel_index, steer_index])

            # step the simulation
            sim.step(dt)
            if initial_warmup > 0:
                initial_warmup -= 1


class WaymoDataset(torch.utils.data.IterableDataset):
    """Waymo dataset loader."""

    def __init__(self, data_path, dataloader_config={}, scenario_config={}, file_limit=None):
        super(WaymoDataset).__init__()

        # save configs
        self.dataloader_config = dataloader_config
        self.scenario_config = scenario_config

        # get paths of dataset files (up to file_limit paths)
        self.file_paths = list(Path(data_path).glob('tfrecord*.json'))[:file_limit]
        print(f'WaymoDataset: loading {len(self.file_paths)} files.')

        # sort the paths for reproducibility if testing on a small set of files
        self.file_paths.sort()

    def __iter__(self):
        """Partition files for each worker and return an (state, expert_action) iterable."""
        # get info on current worker process
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # single-process data loading, return the whole set of files
            return _get_waymo_iterator(self.file_paths, self.dataloader_config, self.scenario_config)

        # distribute a unique set of file paths to each worker process
        worker_file_paths = np.array_split(
            self.file_paths, worker_info.num_workers
        )[worker_info.id]
        return _get_waymo_iterator(list(worker_file_paths), self.dataloader_config, self.scenario_config)


if __name__ == '__main__':
    dataset = WaymoDataset(
        data_path='dataset/tf_records',
        file_limit=20,
        dataloader_config={
            'view_dist': 80,
            'n_stacked_states': 3,
        },
        scenario_config={
            'start_time': 0,
            'allow_non_vehicles': True,
            'spawn_invalid_objects': True,
        }
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
    )

    for i, x in zip(range(100), data_loader):
        print(i, x[0].shape, x[1].shape)
