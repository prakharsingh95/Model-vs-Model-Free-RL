'''Train VAE model on data created using extract.py.'''

import argparse
import os
import settings
import torch

import dataset

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from vae.vae import VAE

cwd = Path(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--savefile', type=str, default=f'{cwd}/vae.pt',
                        help='Location to save VAE model parameters.')
    args = parser.parse_args()
    return args


def run(model, train_loader, test_loader):
    results = cwd/'results'
    results.mkdir(exist_ok=True)
    for epoch in range(settings.num_epochs):
      model.train_on_data(epoch, train_loader)
      model.test_on_data(epoch, test_loader)
      with torch.no_grad():
        sample = torch.randn(64, 32).to(settings.device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 3, 64, 64),
                   f'{results}/sample_{epoch}.png')


def main():
    args = parse_args()
    torch.manual_seed(settings.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if settings.cuda else {}

    train_csv = os.path.join('car_racing', 'train_images.txt')
    test_csv = os.path.join('car_racing', 'test_images.txt')
    train_dataset = dataset.CSVDataset(train_csv)
    test_dataset = dataset.CSVDataset(test_csv)

    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size,
                              shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size,
                             shuffle=True, **kwargs)

    model = VAE(input_size=(3, 64, 64), latent_dim=32).to(settings.device)

    savefile = Path(args.savefile)
    if savefile.exists():
        model.load_state_dict(torch.load(f'{savefile}'))
        model.eval()
    run(model, train_loader, test_loader)
    torch.save(model.state_dict(), f'{savefile}')


if __name__ == '__main__':
    main()
