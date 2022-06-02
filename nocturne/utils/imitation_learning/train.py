"""Imitation learning training script (behavioral cloning)."""
import argparse
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import multiprocessing

from nocturne.utils.imitation_learning.model import ImitationAgent
from nocturne.utils.imitation_learning.waymo_data_loader import WaymoDataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--path', type=str, default='dataset/tf_records',
        help='Path to the training data (directory containing .json scenario files).')
    parser.add_argument('--file_limit', type=int, default=None, help='Limit on the number of files to train on')

    # training
    parser.add_argument('--n_cpus', type=int, default=multiprocessing.cpu_count() - 1,
        help='Number of processes to use for dataset precomputing and loading.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--samples_per_epoch', type=int, default=50000, help='Train batch size')
    parser.add_argument('--batch_size', type=int, default=256, help='Minibatch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training iterations')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')

    # config
    parser.add_argument('--n_stacked_states', type=int, default=5, help='Number of states to stack.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # create dataset
    dataset = WaymoDataset(
        data_path=args.path,
        file_limit=args.file_limit,
        dataloader_config={
            'tmin': 0,
            'tmax': 90,
            'view_dist': 80,
            'view_angle': np.radians(120),
            'dt': 0.1,
            'expert_action_bounds': [[-3, 3], [-0.7, 0.7]],
            'state_normalization': 100,
            'n_stacked_states': args.n_stacked_states,
        },
        scenario_config={
            'start_time': 0,
            'allow_non_vehicles': True,
            'spawn_invalid_objects': True,
            'max_visible_road_points': 300,
            'sample_every_n': 1,
            'road_edge_first': False,
        }
    )

    # create dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.n_cpus,
        pin_memory=True,
    )

    # create model
    n_states = len(dataset[0][0])
    n_actions = len(dataset[0][1])
    model = ImitationAgent(n_states, n_actions, hidden_layers=[1024, 256, 128]).to(args.device)
    model.train()
    print(model)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # create LR scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=args.epochs, verbose=True)

    # train loop
    metrics = []
    for epoch in range(args.epochs):
        print(f'\nepoch {epoch+1}/{args.epochs}')

        losses = []
        l2_norms = []
        for batch, (states, expert_actions) in enumerate(
                tqdm(data_loader, unit='batch')):
            states = states.to(args.device)
            expert_actions = expert_actions.to(args.device)
            dist = model.dist(states)
            # print(dist.mean, dist.variance, expert_actions)
            loss = -dist.log_prob(expert_actions).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            l2_norms.append(
                np.mean(np.linalg.norm(expert_actions.detach().numpy() - dist.mean.detach().numpy(), axis=1)))
        scheduler.step()

        print(f'avg training loss this epoch: {np.mean(losses):.3f}')
        print(f'avg action l2 norm: {np.mean(l2_norms):.3f}')

        # cr = compute_average_collision_rate(eval_trajs, model)
        # ade = compute_average_displacement(eval_trajs, model)
        # grr = compute_average_goal_reaching_rate(eval_trajs, model)
        # print('cr', cr, 'ade', ade, 'grr', grr)

        # metrics.append((np.mean(losses), cr, ade, grr))

        model_path = 'model.pth'
        torch.save(model, model_path)
        print(f'\nSaved model at {model_path}')

    print('metrics', metrics)
