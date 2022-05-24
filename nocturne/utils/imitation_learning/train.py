import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR  # , ExponentialLR
from tqdm import tqdm
import multiprocessing

from cfgs.config import PROCESSED_TRAIN_NO_TL

from nocturne.utils.imitation_learning.model import ImitationAgent
from nocturne.utils.imitation_learning.waymo_data_loader import WaymoDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        type=str,
        default='dataset/json_files',
        help='Path to the training data (directory containing .json scenario '
             'files).',
    )
    parser.add_argument(
        '--precompute',
        action='store_true',
        help='Whether or not to precompute the dataset. This should be run '
             'before the first training or everytime changes in the states '
             'or actions getters are made.'
    )
    parser.add_argument(
        '--n_cpus',
        type=int,
        default=multiprocessing.cpu_count(),
        help='Number of processes to use for dataset precomputing and loading.'
    )
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Minibatch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training iterations')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--file_limit', type=int, default=None, help='Limit on the number of files to use/precompute')
    parser.add_argument('--sample_limit', type=int, default=None, help='Limit on the number of samples to use during training')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # create dataset (and potentially precompute data)
    dataset = WaymoDataset({
        'data_path': args.path,
        'precompute_dataset': args.precompute,
        'n_cpus': args.n_cpus,
        'file_limit': args.file_limit,
        'sample_limit': args.sample_limit,
    })

    # create dataloader
    train_dataloader = DataLoader(
        dataset,
        pin_memory=True,
        shuffle=False,  # DO NOT set shuffle=True as shuffling is already done in the dataloader
        batch_size=args.batch_size,
        num_workers=args.n_cpus,
    )

    # create model
    n_states = len(dataset[0][0])
    n_actions = len(dataset[0][1])
    model = ImitationAgent(n_states, n_actions).to(args.device)
    model.train()

    # create optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # create LR scheduler
    scheduler = LinearLR(optimizer,
                         start_factor=1.0,
                         end_factor=0.1,
                         total_iters=args.epochs,
                         verbose=True)

    # train loop
    avg_losses = []
    for epoch in range(args.epochs):
        print(f'\nepoch {epoch+1}/{args.epochs}')

        losses = []
        for batch, (states, expert_actions) in enumerate(
                tqdm(train_dataloader, unit='batch')):
            states = states.to(args.device)
            expert_actions = expert_actions.to(args.device)
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
