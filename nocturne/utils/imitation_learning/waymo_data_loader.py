"""Dataloader for imitation learning in Nocturne."""
import argparse
from pathlib import Path
import os
import multiprocessing
from multiprocessing import Process

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR  # , ExponentialLR
from tqdm import tqdm

from cfgs.config import PROCESSED_TRAIN_NO_TL
from nocturne import Simulation


def precompute_dataset(scenario_paths, to_path, start_index):
    """Construct a precomputed dataset for fast sampling."""
    # min and max timesteps (max included) that should be used in dataset trajectories
    tmin = 1
    tmax = 90

    # delete files if folder exists already contains some of the form name.dataset.json
    # existing_files = list(precomputed_dataset_path.iterdir())
    # if (len(existing_files)) > 0:
    #     for path in existing_files:
    #         if path.suffixes == ['.dataset', '.txt']:
    #             print(f'Deleting {path}')
    #             path.unlink()

    # go through dataset
    i = 0
    s_nan_count = 0
    a_nan_count = 0
    sample_count = 0
    total_sample_count = 0
    for path in scenario_paths:
        print(path)
        output_strs = []
        f = open(Path(to_path) / f'{i + start_index}.dataset.txt', 'w')

        # create simulation
        sim = Simulation(str(path), start_time=tmin)
        scenario = sim.getScenario()

        # for each time and valid vehicle at that time
        for obj in scenario.getObjectsThatMoved():
            obj.expert_control = True
        for time in range(tmin, tmax):
            for veh in scenario.getVehicles():
                expert_action = scenario.expert_action(veh, time)
                if expert_action is not None:
                    expert_action = expert_action.numpy()
                    sa_nan = False

                    if np.isnan(expert_action).any():
                        a_nan_count += 1
                        sa_nan = True

                    # get state
                    veh_state = np.concatenate(
                        (scenario.ego_state(veh),
                         scenario.flattened_visible_state(veh,
                                                          view_dist=120,
                                                          view_angle=3.14)))

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

                    # pick a file where to save it (pre-shuffle the dataset for faster loading of consecutive chunks)
                    f.write(sa_str + '\n')

                    # append (state, action) string to file
                    output_strs.append(sa_str)
                    sample_count += 1

            # step the simulation
            sim.step(0.1)
        f.close()
        i += 1

    print(f'Finished precomputing dataset, yielding {sample_count} \
            (out of {total_sample_count}) samples.')
    print(
        f'{s_nan_count} samples ({round(100 * s_nan_count / (s_nan_count + sample_count), 1)}%)\
             were ignored because their state contained NaN.')
    print(
        f'{a_nan_count} samples ({round(100 * a_nan_count / (a_nan_count + sample_count), 1)}%)\
             were ignored because their action contained NaN.')


class WaymoDataset(Dataset):
    """Pytorch Dataset for the Waymo data."""

    def __init__(self, cfg):
        """Initialize Dataset."""
        self.cached_data = None
        self.dataset_path = Path(cfg['dataset_path'])

        self.file_paths = list(self.dataset_path.iterdir())
        self.sample_limit = cfg['sample_limit']
        # sort file names
        self.file_paths.sort(key=lambda fp: int(fp.stem.split('.')[0]))

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

        print(
            f'Found {len(self.file_paths)} files for a total of {len(self)} samples.'
        )
        print(
            f'Observation shape is {state_example.shape}, eg: {state_example}')
        print(
            f'Expert action shape is {action_example.shape}, eg: {action_example}'
        )

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


class ImitationAgent(nn.Module):
    """Pytorch Module for imitation. Output is a Multivariable Gaussian."""

    def __init__(self, n_states, n_actions, n_hidden=256):
        """Initialize."""
        super(ImitationAgent, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions

        self.deterministic = False

        self.nn = nn.Sequential(
            nn.Linear(in_features=n_states, out_features=n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=n_hidden, out_features=n_actions, bias=True),
        )

    def dist(self, x):
        """Construct a distirbution from tensor x."""
        x_out = self.nn(x)
        m = MultivariateNormal(x_out[:, 0:2],
                               torch.diag_embed(torch.exp(x_out[:, 2:4])))
        return m

    def forward(self, x):
        """Generate an output from tensor x."""
        m = self.dist(x)
        if self.deterministic:
            return m.mean
        else:
            return m.sample()


if __name__ == '__main__':
    print('\n\nDONT FORGET python setup.py develop\n\n')

    data_path = PROCESSED_TRAIN_NO_TL  # './dataset/json_files'
    data_precomputed_path = './dataset/json_files_precomputed'
    lr = 3e-4
    batch_size = 4096
    n_epochs = 200
    n_workers_dataloader = 8
    device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--precompute', action='store_true'
    )  # Setting this will erase the whole content of the --to folder!
    parser.add_argument('--parallel',
                        action='store_true',
                        help='If true, the precomputation is done in parallel')
    parser.add_argument(
        "--n_processes",
        type=int,
        default=40,
        help="Number of processes over which to split file generation")
    args = parser.parse_args()

    if args.precompute:
        # create folder for precomputed dataset
        precomputed_dataset_path = Path(data_precomputed_path)
        if not os.path.exists(precomputed_dataset_path):
            os.makedirs(str(data_precomputed_path), exist_ok=True)
        # get dataset files
        dataset_path = Path(data_path)
        scenario_paths = list(dataset_path.iterdir())[:1000]
        scenario_paths = [
            file for file in scenario_paths if 'tfrecord' in str(file)
        ]
        if args.parallel:
            # leave some cpus free but have at least one and don't use more than n_processes
            num_cpus = min(max(multiprocessing.cpu_count() - 2, 1),
                           args.n_processes)
            num_files = len(scenario_paths)
            process_list = []
            for i in range(num_cpus):
                p = Process(
                    target=precompute_dataset,
                    args=[
                        scenario_paths[i * num_files // num_cpus:(i + 1) *
                                       num_files // num_cpus],
                        data_precomputed_path, i * num_files // num_cpus
                    ])
                p.start()
                process_list.append(p)

            for process in process_list:
                process.join()
        else:
            precompute_dataset(scenario_paths,
                               data_precomputed_path,
                               start_index=0)

    print(f"Using {device} device")

    print('Initializing dataset...')
    dataset = WaymoDataset({
        'dataset_path': data_precomputed_path,
        'sample_limit': None,  # 100,
    })

    train_dataloader = DataLoader(
        dataset,
        pin_memory=True,
        shuffle=False,
        batch_size=batch_size,
        num_workers=n_workers_dataloader,
    )

    n_states = len(dataset[0][0])
    n_actions = 2 * len(dataset[0][1])
    model = ImitationAgent(n_states, n_actions).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = LinearLR(optimizer,
                         start_factor=1.0,
                         end_factor=0.1,
                         total_iters=n_epochs,
                         verbose=True)

    # train loop
    avg_losses = []
    for epoch in range(n_epochs):
        print(f'\nepoch {epoch+1}/{n_epochs}')

        losses = []
        for batch, (states, expert_actions) in enumerate(
                tqdm(train_dataloader, unit='batch')):
            states = states.to(device)
            expert_actions = expert_actions.to(device)
            dist = model.dist(states)
            # print(dist.mean, dist.variance, expert_actions)
            loss = -dist.log_prob(expert_actions).mean()
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        avg_losses.append(np.mean(losses))
        print(f'avg training loss this epoch: {np.mean(losses):.3f}')

        model_path = 'model.pth'
        torch.save(model, model_path)
        print(f'\nSaved model at {model_path}')

    print('Average losses', avg_losses)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(avg_losses)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.show()
