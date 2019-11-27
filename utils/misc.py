'''Miscellaneous functions and classes.'''

import gym
import numpy as np
import torch

import settings

from pathlib import Path
from torchvision import transforms

from controller import Controller
#from mdrnn import MixtureDensityLSTM
from mdrnn import MixtureDensityLSTMCell
from vae import VAE


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened


def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()


def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)


class RolloutGenerator():
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the
    TRUE ENV using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, device, time_limit):
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
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
        self.time_limit = time_limit

    def get_action_and_transition(self, obs, hidden):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for
        next latent and next hidden state using the MDRNN and compute
        the controller corresponding action.

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        _, latent, _, _ = self.vae(obs)
        action = self.controller(latent, hidden[0])
        _, _, _, _, _, next_hidden = self.mdrnn(action, hidden, latent)
        return action.squeeze().cpu().detach().numpy(), next_hidden

    def rollout(self, params, render=False):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout.
        This is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()

        # This first render is required!
        self.env.render()

        hidden = [
            torch.zeros(1, settings.mdrnn_hidden_dim).to(self.device)
            for _ in range(2)]

        cumulative = 0
        i = 0
        while True:
            obs = transform(obs).unsqueeze(0).to(self.device)
            action, hidden = self.get_action_and_transition(obs, hidden)
            obs, reward, done, _ = self.env.step(action)

            if render:
                self.env.render()

            cumulative += reward
            if done or i > self.time_limit:
                return -cumulative
            i += 1
