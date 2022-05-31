"""Dataloader for imitation learning in Nocturne."""
from multiprocessing import Process
import numpy as np
from pathlib import Path
import torch
from collections import defaultdict

from nocturne import Simulation


# min and max timesteps (max included) that should be used in dataset trajectories
TMIN = 0
TMAX = 89

# view distance and angle used to compute the observations of the agents
VIEW_DIST = 120
VIEW_ANGLE = 3.14


class WaymoDataset(torch.utils.data.Dataset):
    """Pytorch Dataset for the Waymo data."""

    def __init__(self, cfg):
        """Initialize Dataset."""
        # load config
        self.data_path = Path(cfg['data_path'])
        self.precompute_dataset = cfg.get('precompute_dataset', False)
        self.n_cpus = cfg.get('n_cpus', 4)
        self.file_limit = cfg.get('file_limit', None)
        self.sample_limit = cfg.get('sample_limit', None)
        self.shuffle = cfg.get('shuffle', True)

        # get precomputed dataset
        self.precomputed_data_path = Path(str(self.data_path) + '_precomputed')
        if not self.precomputed_data_path.exists() \
                or not self.precomputed_data_path.is_dir() \
                or self.precompute_dataset:
            self._precompute_dataset()

        # get json file paths (sorted by name)
        self.file_paths = list(self.precomputed_data_path.glob('*.shuffled.dataset.txt'))[:self.file_limit]
        self.file_paths.sort(key=lambda fp: list(map(int, fp.stem.split('.')[:2])))

        # initialize cache
        self.cached_data = None

        self.samples_per_file = []

        state_example = None
        action_example = None
        for fp in self.file_paths:
            with open(fp, 'r') as f:
                content = f.readlines()
                self.samples_per_file.append(len(content))

                s0, a0 = self._parse_state_action(content[0])

                if state_example is None or action_example is None:
                    state_example = s0
                    action_example = a0
                else:
                    assert state_example.shape == s0.shape
                    assert action_example.shape == a0.shape

        print(f'Loaded data at {self.data_path}.')
        print(f'Found {len(self.file_paths)} files, {len(self)} samples.')
        print(f'Observation shape is {state_example.shape}')
        print(f'Expert action shape is {action_example.shape}')

    def _parse_state_action(self, sa_str):
        """Convert string from precomputed dataset into state, action pair."""
        s_str, a_str = sa_str.strip().split(';')
        veh_state = np.array(list(map(float, s_str.split(','))))
        expert_action = np.array(list(map(float, a_str.split(','))))
        return veh_state, expert_action

    def __len__(self):
        """See superclass."""
        return sum(self.samples_per_file) if self.sample_limit is None \
            else min(sum(self.samples_per_file), self.sample_limit)

    def __getitem__(self, idx):
        """See superclass."""
        # find file path containing the sample
        i = 0
        while idx >= (n := self.samples_per_file[i]):
            idx -= n
            i += 1
        file_path = self.file_paths[i]

        # load that file in cache (using cache because dataset is pre-shuffled
        # so getitems should be consecutive, ie dont shuffle dataloader)
        if self.cached_data is None or self.cached_data[0] != file_path:
            with open(file_path, 'r') as f:
                self.cached_data = (file_path, f.readlines())

        # get (state, action) from cache
        veh_state, expert_action = self._parse_state_action(
            self.cached_data[1][idx])

        return torch.as_tensor(veh_state,
                               dtype=torch.float32), torch.as_tensor(
                                   expert_action, dtype=torch.float32)

    def _precompute_dataset(self):
        print(f'Precomputing data from {self.data_path} to {self.precomputed_data_path}.')

        # if precomputed dataset already exists or contains files of the
        # form name.dataset.txt, delete them after user confirmation
        if self.precomputed_data_path.exists():
            files_to_delete = []
            if self.precomputed_data_path.is_dir():
                for file in self.precomputed_data_path.iterdir():
                    if '.dataset' in file.suffixes and '.txt' in file.suffixes:
                        files_to_delete.append(file)
            else:
                files_to_delete.append(self.precomputed_data_path)
            if len(files_to_delete) > 0:
                print('The output path for the precomputed dataset '
                      f'({self.precomputed_data_path}) is not empty '
                      'and contains the following files:\n')
                for file in files_to_delete:
                    print(f'\t{file}')
                print(f'\nThe {len(files_to_delete)} files above will be deleted.')
                print('Proceed with the deletion? (yes/no)')
                answer = None
                while answer not in ['yes', 'no']:
                    answer = input()
                if answer == 'no':
                    import sys
                    sys.exit(0)
                for file in files_to_delete:
                    file.unlink()
                print(f'{len(files_to_delete)} files have been deleted.\n')

        # create folder for precomputed dataset
        self.precomputed_data_path.mkdir(exist_ok=True)

        # get dataset files
        scenario_paths = [
            file for file in self.data_path.iterdir()
            if 'tfrecord' in file.stem
        ]
        print(f'Found {len(scenario_paths)} scenario files at {self.data_path}.')
        if self.file_limit is not None and self.file_limit < len(scenario_paths):
            scenario_paths = scenario_paths[:self.file_limit]
            print(f'Only precomputing the first {self.file_limit} files '
                  'because file_limit has been set.')

        # distribute dataset precomputation
        n_cpus = min(self.n_cpus, len(scenario_paths))
        print(f'Precomputing data using {n_cpus} parallel processes.\n')

        def process_idx(process):
            return process * len(scenario_paths) // n_cpus

        # _precompute_dataset_impl(scenario_paths[:10], self.precomputed_data_path, start_index=0, process_idx=1)

        process_list = []
        for i in range(n_cpus):
            p = Process(
                target=_precompute_dataset_impl,
                args=[
                    scenario_paths[process_idx(i):process_idx(i+1)],
                    self.precomputed_data_path,
                    process_idx(i),
                    i + 1,
                ]
            )
            p.start()
            process_list.append(p)

        # wait for precomputation to be done
        for process in process_list:
            process.join()
        print(f'\nDataset precomputation done. Files are written at {self.precomputed_data_path}.\n')

        # now pre-shuffle the data
        print('Now shuffling the data...')
        print('...this will take a few minutes but will save a lot of time during training!\n')

        # get list of all samples (file, line) and shuffle it
        samples = []
        for file in self.precomputed_data_path.iterdir():
            if file.suffixes == ['.dataset', '.txt']:
                with open(file, 'r') as f:
                    lines = f.readlines()
                    file_idx = int(file.stem.split('.')[0])
                    for i in range(len(lines)):
                        samples.append((file_idx, i))
        np.random.shuffle(samples)

        # assign chunks of samples to shuffled output files
        n_samples = len(samples)
        samples_per_file = 10000

        def sample_idx(process):
            return process * n_samples // n_cpus

        process_list = []
        for i in range(n_cpus):
            p = Process(
                target=_shuffle_data_impl,
                args=[
                    samples[sample_idx(i):sample_idx(i+1)],
                    self.precomputed_data_path,
                    samples_per_file,
                    i + 1,
                ]
            )
            p.start()
            process_list.append(p)

        # wait for precomputation to be done
        for process in process_list:
            process.join()
        print('\nShuffling done!\n')


def _precompute_dataset_impl(scenario_paths, to_path, start_index, process_idx):
    """Construct a precomputed dataset for fast sampling."""
    # initializer counters
    s_nan_count = 0
    a_nan_count = 0
    sample_count = 0
    total_sample_count = 0

    # go through scenario paths
    for i, path in enumerate(scenario_paths):
        print(f'({process_idx}) Parsing {path} ({i + 1}/{len(scenario_paths)})')

        # create output file
        output_path = Path(to_path) / f'{start_index + i}.dataset.txt'
        f = open(output_path, 'w')

        # create simulation
        sim = Simulation(str(path), start_time=TMIN, allow_non_vehicles=True, spawn_invalid_objects=True)
        scenario = sim.getScenario()

        # set objects to be expert-controlled
        for obj in scenario.getObjects():
            obj.expert_control = True

        # we're interested in vehicles that moved
        objects_of_interest = [obj for obj in scenario.getVehicles()
                               if obj in scenario.getObjectsThatMoved()]

        # save (state, action) pairs for all objects of interests at all time steps
        data = defaultdict(list)
        for time in range(TMIN, TMAX):
            for obj in objects_of_interest:
                expert_action = scenario.expert_action(obj, time)
                if expert_action is not None:
                    expert_action = expert_action.numpy()
                    sa_nan = False

                    # throw out actions containing nan or too large values
                    if np.isnan(expert_action).any() \
                            or expert_action[0] < -2 or expert_action[0] > 3 \
                            or expert_action[1] < -0.8 or expert_action[1] > 0.8:
                        a_nan_count += 1
                        sa_nan = True

                    # get state
                    veh_state = np.concatenate(
                        (scenario.ego_state(obj),
                         scenario.flattened_visible_state(obj,
                                                          view_dist=VIEW_DIST,
                                                          view_angle=VIEW_ANGLE)))
                    # normalize state
                    veh_state /= 100.0

                    data[obj.id].append((veh_state, np.array([obj.position.x, obj.position.y])))

                    # throw out states containing nan
                    if np.isnan(veh_state).any():
                        s_nan_count += 1
                        sa_nan = True

                    total_sample_count += 1
                    if sa_nan:
                        continue

                    # make sure state and action are 1D arrays
                    assert (len(veh_state.shape) == 1
                            and len(expert_action.shape) == 1)

                    # generate (state, action) string
                    sa_str = ','.join(map(str, veh_state)) + ';' + ','.join(
                        map(str, expert_action))

                    # append (state, action) string to file
                    # f.write(sa_str + '\n')

                    sample_count += 1

            # step the simulation
            sim.step(0.1)

        for vid in data:
            v_data = data[vid]
            n_stacked_input = 1
            for k in range(len(v_data) - n_stacked_input - 1):
                state = []
                for j in range(n_stacked_input):
                    state.append(v_data[k+j][0])
                veh_state = np.concatenate(state)
                if np.isnan(state).any():
                    continue
                pos = v_data[k+n_stacked_input-1][1]
                next_pos = v_data[k+n_stacked_input][1]
                if np.isclose(pos[0], -10000) or np.isclose(next_pos[0], -10000):
                    continue
                expert_action = next_pos - pos
                sa_str = ','.join(map(str, veh_state)) + ';' + ','.join(
                    map(str, expert_action))
                f.write(sa_str + '\n')

        # close output file
        f.close()
        print(f'({process_idx}) Wrote {output_path} ({i + 1}/{len(scenario_paths)})')

    print(f'({process_idx}) Done, precomputed {sample_count} samples out of {total_sample_count} possible '
          f'samples.\n\t{s_nan_count} samples were ignored because their state contained invalid '
          f'values.\n\t{a_nan_count} samples were ignored because their action contained invalid values.',
          flush=True)


def _shuffle_data_impl(samples, data_path, samples_per_file, process_idx):
    current_file_count = 0
    current_file_id = 0
    current_file = open(data_path / f'{current_file_id}.{process_idx}.shuffled.dataset.txt', 'w')
    for i, (sample_file, sample_line) in enumerate(samples):
        if i % 1000 == 0:
            print(f'({process_idx}) {i}/{len(samples)}')
        # read sample line
        with open(data_path / f'{sample_file}.dataset.txt', 'r') as f:
            sample_str = f.readlines()[sample_line]
        # write sample line in output file
        current_file.write(sample_str)
        current_file_count += 1
        if current_file_count >= samples_per_file:
            # open output file
            current_file.close()
            current_file_id += 1
            current_file = open(data_path / f'{current_file_id}.{process_idx}.shuffled.dataset.txt', 'w')
            current_file_count = 0
    current_file.close()


if __name__ == '__main__':
    dataset = WaymoDataset({
        'data_path': './dataset/tf_records',
        'sample_limit': None,  # 100,
        'precompute_dataset': True,
        'n_data_cpus': 4,
        'shuffle': True,
        'file_limit': None,
    })
