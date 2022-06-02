"""Dataloader for imitation learning in Nocturne."""
import torch
from pathlib import Path
import numpy as np

from nocturne import Simulation


def _get_waymo_iterator(paths, tmin=0, tmax=90, view_dist=80, view_angle=120 * 3.14 / 180, dt=0.1,
                        expert_action_bounds=[[-2, 3], [-0.8, 0.8]], state_normalization=100.0,
                        n_stacked_states=5):
    while True:
        # select a random scenario path
        scenario_path = np.random.choice(paths)

        # create simulation
        sim = Simulation(str(scenario_path), start_time=tmin, allow_non_vehicles=True, spawn_invalid_objects=True)
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
                        stacked_state = np.zeros(len(state) * n_stacked_states)
                    stacked_state = np.roll(stacked_state, len(state))
                    stacked_state[:len(state)] = state

                # get expert action
                expert_action = scenario.expert_action(obj, time)
                if expert_action is None:
                    continue
                expert_action = expert_action.numpy()

                # throw out actions containing NaN or out-of-bound values
                if np.isnan(expert_action).any() \
                        or expert_action[0] < expert_action_bounds[0][0] \
                        or expert_action[0] > expert_action_bounds[0][1] \
                        or expert_action[1] < expert_action_bounds[1][0] \
                        or expert_action[1] > expert_action_bounds[1][1]:
                    continue

                # yield state and expert action
                if stacked_state is not None:
                    if initial_warmup <= 0:  # warmup to wait for stacked state to be filled up
                        yield (stacked_state, expert_action)
                else:
                    yield (state, expert_action)

            # step the simulation
            sim.step(dt)
            if initial_warmup > 0:
                initial_warmup -= 1


class WaymoDataset(torch.utils.data.IterableDataset):
    """Waymo dataset loader."""

    def __init__(self, data_path, file_limit=None):
        super(WaymoDataset).__init__()

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
            return _get_waymo_iterator(self.file_paths)

        # distribute a unique set of file paths to each worker process
        worker_file_paths = np.array_split(
            self.file_paths, worker_info.num_workers
        )[worker_info.id]
        return _get_waymo_iterator(list(worker_file_paths))


if __name__ == '__main__':
    dataset = WaymoDataset(
        data_path='dataset/tf_records',
        file_limit=1,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
    )

    for x in data_loader:
        print(x[0].shape, x[1].shape, np.count_nonzero(x[0]))

    # # create dataloader
    # train_dataloader = DataLoader(
    #     dataset,
    #     pin_memory=True,
    #     shuffle=False,  # shuffling is done in the dataloader for faster sampling
    #     batch_size=args.batch_size,
    #     num_workers=args.n_cpus,
    # )
