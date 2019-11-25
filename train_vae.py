#!/usr/bin/env python3
'''Train VAE on data created through extract.py.'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import settings
import torch

import dataset

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from vae.vae import VAE

cwd = Path(os.path.dirname(__file__))
results = cwd/'results'
viz = results/'vae'
viz.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--savefile', type=str, default=f'{cwd}/vae.pt',
                        help='Location to save VAE parameters.')
    args = parser.parse_args()
    return args


def train_on_data(vae, epoch, train_loader):
    vae.train()
    data = train_loader.__iter__()
    length = len(train_loader)
    average_loss = 0
    with tqdm(total=length) as pbar:
        for i in range(1, length+1):
            loss = vae.step(next(data))
            average_loss = (i * average_loss + loss.item()) / (i + 1)
            pbar.set_description(f'Average batch loss {average_loss:.3f}')
            pbar.update(1)


def test_on_data(vae, epoch, test_loader):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            data = data.to(settings.device)
            recon_batch, _, mu, logvar = vae.forward(data)
            test_loss += vae.loss_function(recon_batch, data, mu,
                                           logvar).item()
    return test_loss / len(test_loader)


def plot_sample_space(vae, epoch):
    '''Visualize the VAE distribution.'''
    sample_size = 64
    sample = torch.randn(sample_size, vae.latent_dim).to(settings.device)
    sample = vae.decode(sample).cpu()
    save_image(sample.view(sample_size, vae.channels, vae.img_width,
                           vae.img_height),
               f'{viz}/sample_{epoch}.png')


def plot_autoencoding(vae, dataset, epoch):
    '''Visualize how well the VAE autoencodes.'''
    num_comparisons = 8
    indices = np.random.randint(len(dataset), size=num_comparisons)
    images = np.array([dataset[i] for i in indices])
    tensor = torch.from_numpy(np.array(images)).to(settings.device)
    reconstruction_tensor, _, _, _ = vae(tensor)
    reconstruction = reconstruction_tensor.cpu().data.numpy()
    fig = plt.figure(figsize=(4*num_comparisons, num_comparisons))
    for i in range(num_comparisons):
        plt.subplot(2, num_comparisons, i+1)
        plt.imshow(images[i].transpose((1, 2, 0)))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, num_comparisons, num_comparisons+i+1)
        plt.imshow(reconstruction[i].transpose((1, 2, 0)))
        plt.xticks([])
        plt.yticks([])
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'{viz}/reconstruction_{epoch}.png')


def run(vae, savefile):
    # TODO: Load the train and test images from a single file, and then
    # split
    train_csv = os.path.join('car_racing', 'train_images.txt')
    test_csv  = os.path.join('car_racing', 'test_images.txt')
    train_dataset = dataset.CSVDataset(train_csv)
    test_dataset = dataset.CSVDataset(test_csv)
    kwargs = {'num_workers': 1, 'pin_memory': True} if settings.cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size,
                              shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size,
                             shuffle=True, **kwargs)
    for epoch in range(settings.num_epochs):
        train_on_data(vae, epoch, train_loader)
        loss = test_on_data(vae, epoch, test_loader)
        print(f'====> Test set average batch loss: {loss:.4f}')
        torch.save(vae.state_dict(), f'{savefile}')  # Save after every epoch
        with torch.no_grad():
            plot_sample_space(vae, epoch)
            plot_autoencoding(vae, test_dataset, epoch)


def main():
    args = parse_args()
    input_size = (settings.reduced_image_channels,
                  settings.reduced_image_width,
                  settings.reduced_image_height)
    vae = VAE(input_size=input_size,
              latent_dim=settings.vae_latent_dim).to(settings.device)
    savefile = Path(args.savefile)
    if savefile.exists():
        vae.load_state_dict(torch.load(f'{savefile}'))
        vae.eval()
    run(vae, savefile)


if __name__ == '__main__':
    main()
