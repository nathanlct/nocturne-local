import glob
from pathlib import Path
import os

import numpy as np
from torch.utils.data import Dataset, DataLoader

from nocturne import Simulation


class WaymoDataset(Dataset):
    def __init__(self, cfg):
        file_path = Path(cfg['file_path'])
        self.files = list(file_path.glob('*tfrecord*'))
        # TODO(ev) this is just the number of files, 
        # the actual number of samples is harder to go
        self.num_samples = len(self.files)

    def __len__(self):
        return self.num_samples

    def __get__item(self, idx):
        # construct a scenario
        scenario_path = self.files[idx]
        # sample a start time for the scenario
        start_time = np.random.randint(low=0, high=90)
        sim = Simulation(scenario_path, start_time=start_time)
        vehicles = sim.scenario.getVehicles()
        # not all the vehicles have expert actions at every time-step
        valid_vehs = [veh.getId() for veh in vehicles if sim.scenario.hasExpertAction(veh.getId())]
        veh_id = valid_vehs[np.random.randint(len(vehicles))].getId()
        veh_state = sim.scenario.getState(veh_id)
        expert_action = sim.scenario.getExpertAction(veh_id)
        return veh_state, expert_action

if __name__== '__main__':
    path = '/checkpoint/eugenevinitsky/waymo_open/motion/scenario/training'

    data_loader = DataLoader(WaymoDataset({'file_path': path}),
                            pin_memory=True, shuffle=True, 
                            batch_size=32, num_workers=4)
