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


def train_on_data(model, epoch, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        loss = model.step(data)
        train_loss += loss.item()
        if batch_idx % settings.log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader),
              loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test_on_data(model, epoch, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(settings.device)
            recon_batch, mu, logvar = model.forward(data)
            test_loss = model.loss_function(recon_batch, data, mu,
                                           logvar).item()
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def run(model):
    train_csv = os.path.join('car_racing', 'train_images.txt')
    test_csv = os.path.join('car_racing', 'test_images.txt')
    train_dataset = dataset.CSVDataset(train_csv)
    test_dataset = dataset.CSVDataset(test_csv)
    kwargs = {'num_workers': 1, 'pin_memory': True} if settings.cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size,
                              shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size,
                             shuffle=True, **kwargs)

    results = cwd/'results'
    results.mkdir(exist_ok=True)
    for epoch in range(settings.num_epochs):
        train_on_data(model, epoch, train_loader)
        test_on_data(model, epoch, test_loader)
        with torch.no_grad():
            # TODO: Add autoencoder comparison
            sample = torch.randn(64, 32).to(settings.device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, 64, 64),
                       f'{results}/sample_{epoch}.png')


def main():
    args = parse_args()
    torch.manual_seed(settings.seed)

    model = VAE(input_size=(3, 64, 64), latent_dim=32).to(settings.device)
    savefile = Path(args.savefile)
    if savefile.exists():
        model.load_state_dict(torch.load(f'{savefile}'))
        model.eval()
    run(model)
    torch.save(model.state_dict(), f'{savefile}')


if __name__ == '__main__':
    main()
