"""Implements a Mixture Density LSTM Network for predicting the future latents 
based on past latents
"""

import torch
import torch.nn as nn
import torch.nn.functional as f


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
        self.muLayer = nn.Linear(
            self.NUM_HIDDENS, self.NUM_LATENTS * self.NUM_GAUSSIANS)
        self.sigmaLayer = nn.Linear(
            self.NUM_HIDDENS, self.NUM_LATENTS * self.NUM_GAUSSIANS)
        self.MDLogProbLayer = nn.Linear(self.NUM_HIDDENS, self.NUM_GAUSSIANS)
        self.rewardLayer = nn.Linear(self.NUM_HIDDENS, 1)
        self.terminationProbLayer = nn.Linear(self.NUM_HIDDENS, 1)

    def forward(self, x):
        pass


class MixtureDensityLSTM(_MixtureDensityLSTMBase):

    def __init__(self, NUM_LATENTS: int, NUM_ACTIONS: int,
                 NUM_HIDDENS: int, NUM_GAUSSIANS: int):
        super(MixtureDensityLSTM, self).__init__(
            NUM_LATENTS, NUM_ACTIONS, NUM_HIDDENS, NUM_GAUSSIANS)

        self.lstm = nn.LSTM(self.NUM_LATENTS + self.NUM_ACTIONS,
                            self.NUM_HIDDENS, num_layers=1, batch_first=True)

    def forward(self,  inActions, inLatents):

        BATCH_SIZE, SEQ_LEN = inActions.shape[0], inActions.shape[1]

        inputs = torch.cat((inActions, inLatents), dim=-1)

        # h0 and c0 are treated as 0, which would not be the case during 
        # inference
        outLSTMStates, _ = self.lstm(inputs)


        # Means of output latents
        outMus = self.muLayer(outLSTMStates)
        outMus = outMus.view(BATCH_SIZE, SEQ_LEN, self.NUM_GAUSSIANS, self.NUM_LATENTS)

        # Sigmas of output latents
        outSigmas = self.sigmaLayer(outLSTMStates)
        outSigmas = outSigmas.view(
            BATCH_SIZE, SEQ_LEN, self.NUM_GAUSSIANS, self.NUM_LATENTS)
        outSigmas = outSigmas.exp()

        # Log probabilities
        outMDLogProbs = self.MDLogProbLayer(outLSTMStates)
        outMDLogProbs = outMDLogProbs.view(BATCH_SIZE, SEQ_LEN, self.NUM_GAUSSIANS)
        outMDLogProbs = f.log_softmax(outMDLogProbs, dim=-1)

        # Reward
        outRewards = self.rewardLayer(outLSTMStates)

        # Episode termination probability
        outTerminationProbs = self.terminationProbLayer(outLSTMStates)
        outTerminationProbs = torch.sigmoid(outTerminationProbs)

        return outMus, outSigmas, outMDLogProbs, outRewards, outTerminationProbs


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
        self.lstmCell = nn.LSTMCell(
            self.NUM_LATENTS + self.NUM_ACTIONS, self.NUM_HIDDENS)

    def forward(self, inActions, in_hidden, inLatents):
        """Predict the next latent distribution given the previous 
        latent distribution

        Args:
            inActions: Action to predict for
            in_hidden: Previous LSTM state
            inLatents: Previous latents (this is NOT a distribution)

        Returns:
            5-tuple: outMus, outSigmas, outMDLogProbs, outRewards, 
            outTerminationProbs
            Returns means, sigmas and mixtures for the next latent, along 
            with predictions for the next reward and whether the episode 
            will terminate or not
        """
        inputs = torch.cat((inActions, inLatents), dim=-1)
        out_hiddens = self.lstmCell(inputs, in_hidden)
        out_states = out_hiddens[0]

        BATCH_SIZE = inActions.shape[0]

        # Means of output latents
        outMus = self.muLayer(out_states)
        outMus = outMus.view(BATCH_SIZE, self.NUM_GAUSSIANS, self.NUM_LATENTS)

        # Sigmas of output latents
        outSigmas = self.sigmaLayer(out_states)
        outSigmas = outSigmas.view(
            BATCH_SIZE, self.NUM_GAUSSIANS, self.NUM_LATENTS)
        outSigmas = outSigmas.exp()

        # Log probabilities
        outMDLogProbs = self.MDLogProbLayer(out_states)
        outMDLogProbs = outMDLogProbs.view(BATCH_SIZE, self.NUM_GAUSSIANS)
        outMDLogProbs = f.log_softmax(outMDLogProbs, dim=-1)

        # Reward
        outRewards = self.rewardLayer(out_states)

        # Episode termination probability
        outTerminationProbs = self.terminationProbLayer(out_states)
        outTerminationProbs = torch.sigmoid(outTerminationProbs)

        return outMus, outSigmas, outMDLogProbs, outRewards,
        outTerminationProbs, out_hiddens
