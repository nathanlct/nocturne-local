"""Imitation learning training script (behavioral cloning)."""
import argparse
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import multiprocessing
import json
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader

from nocturne.utils.imitation_learning.model import ImitationAgent
from nocturne.utils.imitation_learning.waymo_data_loader import WaymoDataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--path',
        type=str,
        default='dataset/tf_records',
        help=
        'Path to the training data (directory containing .json scenario files).'
    )
    parser.add_argument('--file_limit',
                        type=int,
                        default=None,
                        help='Limit on the number of files to train on')

    # training
    parser.add_argument(
        '--n_cpus',
        type=int,
        default=multiprocessing.cpu_count() - 1,
        help='Number of processes to use for dataset precomputing and loading.'
    )
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--samples_per_epoch',
                        type=int,
                        default=50000,
                        help='Train batch size')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Minibatch size')
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='Number of training iterations')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--discrete',
                        action='store_true',
                        help='If true use a discrete action space')

    # config
    parser.add_argument('--n_stacked_states',
                        type=int,
                        default=10,
                        help='Number of states to stack.')
    parser.add_argument('--view_dist',
                        type=float,
                        default=80,
                        help='Visible state view distance.')
    parser.add_argument('--view_angle',
                        type=float,
                        default=np.radians(120),
                        help='Visible state cone angle.')
    parser.add_argument(
        '--actions_are_positions',
        action='store_true',
        help='If true the model outputs diff in position and heading directly')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # create dataset and dataloader
    if args.actions_are_positions:
        expert_bounds = [[-0.5, 3], [-3, 3], [-0.07, 0.07]]
        actions_discretizations = [21, 21, 21]
        actions_bounds = [[-0.5, 3], [-3, 3], [-0.07, 0.07]]
        mean_scalings = [3, 3, 0.07]
        std_devs = [0.1, 0.1, 0.02]
    else:
        expert_bounds = [[-3, 3], [-0.7, 0.7]]
        actions_bounds = [[-3, 3], [-0.7, 0.7]]
        actions_discretizations = [7, 21]
        mean_scalings = [3, 0.7]
        std_devs = [0.1, 0.02]

    dataloader_cfg = {
        'tmin': 0,
        'tmax': 90,
        'view_dist': args.view_dist,
        'view_angle': args.view_angle,
        'dt': 0.1,
        'expert_action_bounds': expert_bounds,
        'expert_position': args.actions_are_positions,
        'state_normalization': 100,
        'n_stacked_states': args.n_stacked_states,
    }
    scenario_cfg = {
        'start_time': 0,
        'allow_non_vehicles': True,
        'spawn_invalid_objects': True,
        'max_visible_road_points': 1000,
        'sample_every_n': 1,
        'road_edge_first': False,
    }
    dataset = WaymoDataset(
        data_path=args.path,
        file_limit=args.file_limit,
        dataloader_config=dataloader_cfg,
        scenario_config=scenario_cfg,
    )
    data_loader = iter(
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.n_cpus,
            pin_memory=True,
        ))

    # create model
    sample_state, _ = next(data_loader)
    n_states = sample_state.shape[-1]

    model_cfg = {
        'n_inputs': n_states,
        'hidden_layers': [1024, 256, 128],
        'discrete': args.discrete,
        'mean_scalings': mean_scalings,
        'std_devs': std_devs,
        'actions_discretizations': actions_discretizations,
        'actions_bounds': actions_bounds,
        'device': args.device
    }

    model = ImitationAgent(model_cfg).to(args.device)
    model.train()
    print(model)

    # create optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # create exp dir
    time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    exp_dir = Path('train_logs') / time_str
    exp_dir.mkdir(parents=True, exist_ok=True)

    # save configs
    configs_path = exp_dir / 'configs.json'
    configs = {
        'scenario_cfg': scenario_cfg,
        'dataloader_cfg': dataloader_cfg,
        'model_cfg': model_cfg,
    }
    with open(configs_path, 'w') as fp:
        json.dump(configs, fp, sort_keys=True, indent=4)
    print('Wrote configs at', configs_path)

    # tensorboard writer
    writer = SummaryWriter(log_dir=str(exp_dir))

    # train loop
    print('Exp dir created at', exp_dir)
    print(f'`tensorboard --logdir={exp_dir}`\n')
    for epoch in range(args.epochs):
        print(f'\nepoch {epoch+1}/{args.epochs}')
        n_samples = epoch * args.batch_size * (args.samples_per_epoch //
                                               args.batch_size)

        for i in tqdm(range(args.samples_per_epoch // args.batch_size),
                      unit='batch'):
            # get states and expert actions
            states, expert_actions = next(data_loader)
            states = states.to(args.device)
            expert_actions = expert_actions.to(args.device)

            # compute loss
            if args.discrete:
                log_prob, expert_idxs = model.log_prob(states,
                                                       expert_actions,
                                                       return_indexes=True)
            else:
                dist = model.dist(states)
                log_prob = dist.log_prob(expert_actions.float())
            loss = -log_prob.mean()

            # optim step
            optimizer.zero_grad()
            loss.backward()

            # grad clipping
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item()**2
            total_norm = total_norm**0.5
            writer.add_scalar('train/grad_norm', total_norm, n_samples)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item()**2
            total_norm = total_norm**0.5
            writer.add_scalar('train/post_clip_grad_norm', total_norm,
                              n_samples)
            optimizer.step()

            # tensorboard logging
            writer.add_scalar('train/loss', loss.item(), n_samples)

            if args.actions_are_positions:
                writer.add_scalar('train/x_logprob', log_prob[0], n_samples)
                writer.add_scalar('train/y_logprob', log_prob[1], n_samples)
                writer.add_scalar('train/steer_logprob', log_prob[2],
                                  n_samples)
            else:
                writer.add_scalar('train/accel_logprob', log_prob[0],
                                  n_samples)
                writer.add_scalar('train/steer_logprob', log_prob[1],
                                  n_samples)

            if not model_cfg['discrete']:
                diff_actions = torch.mean(torch.abs(dist.mean -
                                                    expert_actions),
                                          axis=0)
                writer.add_scalar('train/accel_diff', diff_actions[0],
                                  n_samples)
                writer.add_scalar('train/steer_diff', diff_actions[1],
                                  n_samples)
                writer.add_scalar(
                    'train/l2_dist',
                    torch.norm(dist.mean - expert_actions.float()), n_samples)

            if model_cfg['discrete']:
                with torch.no_grad():
                    model_actions, model_idxs = model(states,
                                                      deterministic=True,
                                                      return_indexes=True)
                accuracy = [
                    (model_idx == expert_idx).float().mean(axis=0)
                    for model_idx, expert_idx in zip(model_idxs, expert_idxs.T)
                ]
                if args.actions_are_positions:
                    writer.add_scalar('train/x_pos_acc', accuracy[0],
                                      n_samples)
                    writer.add_scalar('train/y_pos_acc', accuracy[1],
                                      n_samples)
                    writer.add_scalar('train/heading_acc', accuracy[2],
                                      n_samples)
                else:
                    writer.add_scalar('train/accel_acc', accuracy[0],
                                      n_samples)
                    writer.add_scalar('train/steer_acc', accuracy[1],
                                      n_samples)

        # save model checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            model_path = exp_dir / f'model_{epoch+1}.pth'
            torch.save(model, str(model_path))
            pickle.dump(filter, open(exp_dir / f"filter_{epoch+1}.pth", "wb"))
            print(f'\nSaved model at {model_path}')
        if args.discrete:
            if args.actions_are_positions:
                print('xpos')
                print('model: ', model_idxs[0][0:10])
                print('expert: ', expert_idxs[0:10, 0])
                print('ypos')
                print('model: ', model_idxs[1][0:10])
                print('expert: ', expert_idxs[0:10, 1])
                print('steer')
                print('model: ', model_idxs[2][0:10])
                print('expert: ', expert_idxs[0:10, 2])
            else:
                print('accel')
                print('model: ', model_idxs[0][0:10])
                print('expert: ', expert_idxs[0:10, 0])
                print('steer')
                print('model: ', model_idxs[1][0:10])
                print('expert: ', expert_idxs[0:10, 1])

    print('Done, exp dir is', exp_dir)

    writer.flush()
    writer.close()
