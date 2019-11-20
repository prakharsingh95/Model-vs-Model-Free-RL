'''Implement the VAE from the paper 'World Models'.'''

import argparse
import os
import settings
import torch
import torch.utils.data

from pathlib import Path
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4,
                               stride=2) # -> 32 x 31 x 31
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                               stride=2) # -> 64 x 14 x 14
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,
                               stride=2) # -> 128 x 6 x 6
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=4, stride=2) # -> 256 x 2 x 2
        compressed_size = 256 * 2 * 2
        self.z_mean = nn.Linear(compressed_size, latent_dim)
        self.z_logsd = nn.Linear(compressed_size, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.shape[0], -1)
        mean = self.z_mean(x)
        logsd = self.z_logsd(x)
        return mean, logsd


class Decoder(nn.Module):
    def __init__(self, output_size, latent_dim):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=128,
                                          kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                          kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                          kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=3,
                                          kernel_size=6, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = x.view(-1, x.shape[1], 1, 1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = x.view(-1, *self.output_size)
        x = self.sigmoid(x)
        return x


class VAE(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(output_size=self.input_size, latent_dim=latent_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    # TODO: Move this to a loss file
    def loss_function(self, recon_x, x, mu, logvar):
        mse = F.mse_loss(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld

    def step(self, data):
        data = data.to(settings.device)
        self.optimizer.zero_grad()
        recon_batch, mu, logvar = self.forward(data)
        loss = self.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss = loss.item()
        self.optimizer.step()
        return loss

    # TODO: Move train and test to train_vae.py
    def train_on_data(self, epoch, train_loader):
        self.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            loss = self.step(data)
            train_loss += loss.item()
            if batch_idx % settings.log_interval == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader),
                  loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))

    def test_on_data(self, epoch, test_loader):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = data.to(settings.device)
                recon_batch, mu, logvar = self.forward(data)
                test_loss = self.loss_function(recon_batch, data, mu,
                                               logvar).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n], recon_batch.view(
                                                          settings.batch_size, 3,
                                                          64, 64)[:n]])
                    save_image(comparison.cpu(),
                               f'reconstruction_{epoch}.png', nrow=n)
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
