#!/usr/bin/env python3
'''Train VAE model on data created using extract.py. Final model saved
into tf_vae/vae.json
'''

import argparse
import os
import torch

import dataset

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from vae.vae import VAE

cwd = Path(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--savefile', type=str, default=f'{cwd}/vae.pt',
                        help='Location to save VAE model parameters.')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def run(model, train_loader, test_loader, args):
    results = cwd/'results'
    results.mkdir(exist_ok=True)
    for epoch in range(args.epochs):
      model.train_on_data(epoch, train_loader, args)
      model.test_on_data(epoch, test_loader, args)
      with torch.no_grad():
        device = torch.device('cuda')
        sample = torch.randn(64, 32).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 3, 64, 64),
                   f'{results}/sample_{epoch}.png')


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_csv = 'train_images.txt'
    test_csv = 'test_images.txt'
    train_dataset = dataset.CarRacingDataset(train_csv, cwd)
    test_dataset = dataset.CarRacingDataset(test_csv, cwd)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=True, **kwargs)

    model = VAE(input_size=(3, 64, 64), latent_dim=32).to(args.device)

    savefile = Path(args.savefile)
    if savefile.exists():
        model.load_state_dict(torch.load(f'{savefile}'))
        model.eval()
    run(model, train_loader, test_loader, args)
    torch.save(model.state_dict(), f'{savefile}')


if __name__ == '__main__':
    main()
