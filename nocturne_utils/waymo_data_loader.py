from pathlib import Path
import time
import os

import numpy as np
from torch.utils.data import Dataset, DataLoader

from nocturne import Simulation


class WaymoDataset(Dataset):
    def __init__(self, cfg):
        file_path = Path(cfg['file_path'])
        self.file_path = file_path
        self.files = os.listdir(file_path) #list(file_path.glob('*tfrecord*'))
        # TODO(ev) this is just the number of files, 
        # the actual number of samples is harder to go
        self.num_samples = len(self.files)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        print('calling get item')
        t = time.time()
        # construct a scenario
        scenario_path = os.path.join(self.file_path, self.files[idx])
        # sample a start time for the scenario (need a non-zero time for now to have expert actions)
        start_time = np.random.randint(low=1, high=90)

        sim = Simulation(str(scenario_path), start_time=start_time)
        scenario = sim.getScenario()
        vehicles = scenario.getVehicles()
        # not all the vehicles have expert actions at every time-step
        valid_vehs = [veh for veh in vehicles if scenario.hasExpertAction(veh.getID(), start_time)]
        int_val = np.random.randint(low=0, high=len(valid_vehs))
        veh_id = valid_vehs[int_val].getID()
        # TODO(ev) put this in when complete
        # veh_state = sim.scenario.getState(veh_id)
        print('made it to get visible state')
        veh_state = scenario.getVisibleObjectsState(valid_vehs[int_val], 1.58)
        expert_action = np.array(scenario.getExpertAction(veh_id, start_time))
        print('returning from get item')
        return veh_state, expert_action

# TODO(ev) move this out of this file
def form_imitation_loss(policy, dataloader):
    states, actions = next(iter(dataloader))
    kl_loss = policy(states).kl(actions)
    return kl_loss


if __name__== '__main__':

    os.environ["DISPLAY"] = ":0.0"
    path = '/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/uncompressed/scenario/formatted_json/'

    num_iters = 1
    data_loader = DataLoader(WaymoDataset({'file_path': path}),
                            pin_memory=True, shuffle=True, 
                            batch_size=32, num_workers=0)
    t = time.time()
    for i, (states, actions) in enumerate(data_loader):
        print(i)
        if i > num_iters:
            break
    print(f'time to generate {num_iters} batches with 1 worker is {time.time() - t}')

    data_loader = DataLoader(WaymoDataset({'file_path': path}),
                            pin_memory=True, shuffle=True, 
                            batch_size=32, num_workers=min(num_iters, 10))
    t = time.time()
    for i, (states, actions) in enumerate(data_loader):
        print(i)
        if i > num_iters:
            break
    print(f'time to generate {num_iters} batches with 4 workers is {time.time() - t}')