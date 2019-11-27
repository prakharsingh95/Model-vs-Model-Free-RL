"""Implements a Mixture Density LSTM Network for predicting the future latents 
based on past latents
"""

import torch
import torch.nn as nn
import torch.nn.functional as f

from torch.distributions.normal import Normal


class _MixtureDensityLSTMBase(nn.Module):

    def __init__(self, NUM_LATENTS: int, NUM_ACTIONS: int,
                 NUM_HIDDENS: int, NUM_GAUSSIANS: int):

        super(_MixtureDensityLSTMBase, self).__init__()

        # Constants
        self.NUM_LATENTS = NUM_LATENTS
        self.NUM_ACTIONS = NUM_ACTIONS
        self.NUM_HIDDENS = NUM_HIDDENS
        self.NUM_GAUSSIANS = NUM_GAUSSIANS

        # Torch layers for forward()
        self.mu_layer = nn.Linear(
            self.NUM_HIDDENS, self.NUM_LATENTS * self.NUM_GAUSSIANS)
        self.sigma_layer = nn.Linear(
            self.NUM_HIDDENS, self.NUM_LATENTS * self.NUM_GAUSSIANS)
        self.md_log_prob_layer = nn.Linear(self.NUM_HIDDENS, self.NUM_GAUSSIANS)
        self.reward_layer = nn.Linear(self.NUM_HIDDENS, 1)
        self.termination_prob_layer = nn.Linear(self.NUM_HIDDENS, 1)

    def forward(self, x):
        pass


class MixtureDensityLSTM(_MixtureDensityLSTMBase):

    def __init__(self, NUM_LATENTS: int, NUM_ACTIONS: int,
                 NUM_HIDDENS: int, NUM_GAUSSIANS: int):
        super(MixtureDensityLSTM, self).__init__(
            NUM_LATENTS, NUM_ACTIONS, NUM_HIDDENS, NUM_GAUSSIANS)

        self.lstm = nn.LSTM(self.NUM_LATENTS + self.NUM_ACTIONS,
                            self.NUM_HIDDENS, num_layers=1, batch_first=True)

    def forward(self,  in_actions, in_latents, sample=False):

        BATCH_SIZE, SEQ_LEN = in_actions.shape[0], in_actions.shape[1]

        inputs = torch.cat((in_actions, in_latents), dim=-1)

        # h0 and c0 are treated as 0, which would not be the case during 
        # inference
        outLSTMStates, _ = self.lstm(inputs)


        # Means of output latents
        out_mus = self.mu_layer(outLSTMStates)
        out_mus = out_mus.view(BATCH_SIZE, SEQ_LEN, self.NUM_GAUSSIANS, self.NUM_LATENTS)

        # Sigmas of output latents
        out_sigmas = self.sigma_layer(outLSTMStates)
        out_sigmas = out_sigmas.view(
            BATCH_SIZE, SEQ_LEN, self.NUM_GAUSSIANS, self.NUM_LATENTS)
        out_sigmas = out_sigmas.exp()

        # Log probabilities
        out_md_logprobs = self.md_log_prob_layer(outLSTMStates)
        out_md_logprobs = out_md_logprobs.view(BATCH_SIZE, SEQ_LEN, self.NUM_GAUSSIANS)
        out_md_logprobs = f.log_softmax(out_md_logprobs, dim=-1)

        # Reward
        out_rewards = self.reward_layer(outLSTMStates)

        # Episode termination probability
        out_termination_probs = self.termination_prob_layer(outLSTMStates)
        out_termination_probs = torch.sigmoid(out_termination_probs)

        if sample:
            distrib = Normal(out_mus, out_sigmas)
            sampled_z_gauss = distrib.rsample()
            out_md_probs = out_md_logprobs.exp().unsqueeze(-1)
            sampled_z = torch.sum(sampled_z_gauss * out_md_probs, dim=-2)
        else:
            sampled_z = None

        return out_mus, out_sigmas, out_md_logprobs, out_rewards, out_termination_probs, sampled_z


class MixtureDensityLSTMCell(_MixtureDensityLSTMBase):

    """Torch nn.Module that implements an MDRNN cell
    """

    def __init__(self, NUM_LATENTS: int, NUM_ACTIONS: int,
                 NUM_HIDDENS: int, NUM_GAUSSIANS: int):
        """Constructor to set up all the relevant torch layers

        Args:
            NUM_LATENTS (int): Size of the latent dimension
            NUM_ACTIONS (int): Number of possible actions
            NUM_HIDDENS (int): Size of the LSTM's hidden layer
            NUM_GAUSSIANS (int): Number of gaussians to use in the mixture model
        """

        super(MixtureDensityLSTMCell, self).__init__(
            NUM_LATENTS, NUM_ACTIONS, NUM_HIDDENS, NUM_GAUSSIANS)

        # Torch LSTM Cell for forward()
        self.lstm_cell = nn.LSTMCell(
            self.NUM_LATENTS + self.NUM_ACTIONS, self.NUM_HIDDENS)

    def forward(self, in_actions, in_hidden, in_latents):
        """Predict the next latent distribution given the previous 
        latent distribution

        Args:
            in_actions: Action to predict for
            in_hidden: Previous LSTM state
            in_latents: Previous latents (this is NOT a distribution)

        Returns:
            5-tuple: out_mus, out_sigmas, out_md_logprobs, out_rewards, 
            out_termination_probs
            Returns means, sigmas and mixtures for the next latent, along 
            with predictions for the next reward and whether the episode 
            will terminate or not
        """
        inputs = torch.cat((in_actions, in_latents), dim=-1)
        out_hiddens = self.lstm_cell(inputs, in_hidden)
        out_states = out_hiddens[0]

        BATCH_SIZE = in_actions.shape[0]

        # Means of output latents
        out_mus = self.mu_layer(out_states)
        out_mus = out_mus.view(BATCH_SIZE, self.NUM_GAUSSIANS, self.NUM_LATENTS)

        # Sigmas of output latents
        out_sigmas = self.sigma_layer(out_states)
        out_sigmas = out_sigmas.view(
            BATCH_SIZE, self.NUM_GAUSSIANS, self.NUM_LATENTS)
        out_sigmas = out_sigmas.exp()

        # Log probabilities
        out_md_logprobs = self.md_log_prob_layer(out_states)
        out_md_logprobs = out_md_logprobs.view(BATCH_SIZE, self.NUM_GAUSSIANS)
        out_md_logprobs = f.log_softmax(out_md_logprobs, dim=-1)

        # Reward
        out_rewards = self.reward_layer(out_states)

        # Episode termination probability
        out_termination_probs = self.termination_prob_layer(out_states)
        out_termination_probs = torch.sigmoid(out_termination_probs)

        return out_mus, out_sigmas, out_md_logprobs, out_rewards, out_termination_probs, out_hiddens
