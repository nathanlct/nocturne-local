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

    def __getitem__(self, idx):
        # TODO(ev) there might not be a valid vehicle?
        # construct a scenario
        scenario_path = self.files[idx]
        # sample a start time for the scenario
        start_time = np.random.randint(low=0, high=90)

        sim = Simulation(str(scenario_path), start_time=start_time)
        scenario = sim.getScenario()
        vehicles = scenario.getVehicles()
        # not all the vehicles have expert actions at every time-step
        valid_vehs = [veh for veh in vehicles if scenario.hasExpertAction(veh.getID(), start_time)]
        veh_id = valid_vehs[np.random.randint(low=0, high=len(valid_vehs))].getID()
        # TODO(ev) put this in when complete
        # veh_state = sim.scenario.getState(veh_id)
        veh_state = np.zeros(2)
        expert_action = np.array(scenario.getExpertAction(veh_id, start_time))
        return veh_state, expert_action

# TODO(ev) move this out of this file
def form_imitation_loss(policy, dataloader):
    states, actions = next(iter(dataloader))
    kl_loss = policy(states).kl(actions)
    return kl_loss


if __name__== '__main__':
    path = '/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/uncompressed/scenario/formatted_json/'

    data_loader = DataLoader(WaymoDataset({'file_path': path}),
                            pin_memory=True, shuffle=True, 
                            batch_size=32, num_workers=0)
    t = time.time()
    states, actions = next(iter(data_loader))
    print(f'time to generate batch with 1 worker is {time.time() - t}')

    # t = time.time()
    # data_loader = DataLoader(WaymoDataset({'file_path': path}),
    #                         pin_memory=True, shuffle=True, 
    #                         batch_size=32, num_workers=4)
    # states, actions = next(iter(data_loader))
    # print(f'time to generate batch with 4 workers is {time.time() - t}')