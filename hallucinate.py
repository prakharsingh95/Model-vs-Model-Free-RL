#!/usr/bin/env python3

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import settings
import utils.misc as misc

from pathlib import Path
from torch.distributions.categorical import Categorical

from controller import Controller
from mdrnn import MixtureDensityLSTMCell
from vae import VAE


class Hallucination(gym.Env):
    def __init__(self, mdir):
        # Pretty much copy-paste from Rollout Generator initialization
        self.action_space = gym.spaces.Box(np.array([-1, 0, 0]),
                                           np.array([1, 1, 1]))
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(64, 64, 3),
                                                dtype=np.uint8)

        device = settings.device
        vae_input_size = (settings.reduced_image_channels,
                          settings.reduced_image_width,
                          settings.reduced_image_height)
        self.vae = VAE(input_size=vae_input_size,
                       latent_dim=settings.vae_latent_dim).to(device)
        vae_savefile =  mdir/'vae.pt'
        self.vae.load_state_dict(torch.load(vae_savefile))
        self.vae.eval()

        self.mdrnn = MixtureDensityLSTMCell(
            settings.vae_latent_dim,
            settings.action_space_size,
            settings.mdrnn_hidden_dim,
            settings.mdrnn_num_gaussians).to(device)

        mdrnn_savefile = mdir/'mdrnn.pt'
        state = torch.load(mdrnn_savefile)
        new_state = {}
        for k, v in state.items():
            new_k = k.rstrip('_l0')
            new_k = new_k.replace('lstm', 'lstm_cell')
            new_state[new_k] = v
        self.mdrnn.load_state_dict(new_state)
        self.mdrnn.eval()

        input_size = (settings.vae_latent_dim
                      + settings.mdrnn_hidden_dim)
        output_size = 3
        self.controller = Controller(input_size, output_size).to(device)

        controller_savefille = mdir/'controller.pt'
        if Path(controller_savefille).exists():
            self.controller.load_state_dict(torch.load(controller_savefille))
        self.controller.eval()

        self.env = gym.make('CarRacing-v0')
        self.device = device
        self.time_limit = 1000

        self.latent = torch.randn(1, self.vae.latent_dim).to(self.device)
        # Multiply by two because have current, and instinctual next-state
        # prediction
        self.hidden = [
            torch.zeros(1, settings.mdrnn_hidden_dim).to(self.device)
            for _ in range(2)]

        self.obs = None
        self.visual_obs = None

        self.monitor = None
        self.figure = None

    def reset(self):
        self.latent = torch.randn(1, self.vae.latent_dim).to(self.device)
        self.hidden = [
            torch.zeros(1, settings.mdrnn_hidden_dim).to(self.device)
            for _ in range(2)]

        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(np.zeros((64, 64, 3), dtype=np.uint8))

    def step(self):
        with torch.no_grad():
            action = self.controller(self.latent, self.hidden[0])

            mu, _, pi, _, _, n_h = self.mdrnn(action, self.hidden, self.latent)
            pi = pi.squeeze()
            mixt = Categorical(torch.exp(pi)).sample().item()
            self.latent = mu[:, mixt, :]
            self.hidden = n_h

            obs = self.vae.decoder(self.latent)
            img = obs.cpu().numpy()
            self.image = img.squeeze().transpose((1, 2, 0))

    def render(self):
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(np.zeros((64, 64, 3)))
        self.monitor.set_data(self.image)
        # This call to pause updates the figure and displays it at a
        # rate of 60 FPS
        plt.pause(1/60)


def main():
    hallucination = Hallucination(mdir=Path('.'))
    hallucination.reset()

    seq_len = 1000

    for _ in range(seq_len):
        hallucination.step()
        hallucination.render()


if __name__ == '__main__':
    main()
