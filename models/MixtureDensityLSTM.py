"""Implements a Mixture Density LSTM Network for predicting the future latents based on past latents
"""

import torch
import torch.nn as nn
import torch.nn.functional as f

class MixtureDensityLSTM(nn.Module):

    """Torch nn.Module that implements an MDRNN network
    """
    
    def __init__(self, NUM_LATENTS:int, NUM_ACTIONS:int, NUM_HIDDENS:int, NUM_GAUSSIANS:int):
        """Constructor to set up all the relevant torch layers
        
        Args:
            NUM_LATENTS (int): Size of the latent dimension
            NUM_ACTIONS (int): Number of possible actions
            NUM_HIDDENS (int): Size of the LSTM's hidden layer
            NUM_GAUSSIANS (int): Number of gaussians to use in the mixture model
        """
        
        super(MixtureDensityLSTM, self).__init__()

        # Constants
        self.NUM_LATENTS = NUM_LATENTS
        self.NUM_ACTIONS = NUM_ACTIONS
        self.NUM_HIDDENS = NUM_HIDDENS
        self.NUM_GAUSSIANS = NUM_GAUSSIANS

        # Torch layers for forward()
        self.muLayer = nn.Linear(self.NUM_HIDDENS, self.NUM_LATENTS * self.NUM_GAUSSIANS)
        self.sigmaLayer = nn.Linear(self.NUM_HIDDENS, self.NUM_LATENTS * self.NUM_GAUSSIANS)
        self.MDLogProbLayer = nn.Linear(self.NUM_HIDDENS, self.NUM_GAUSSIANS)
        self.rewardLayer = nn.Linear(self.NUM_HIDDENS, 1)
        self.terminationProbLayer = nn.Linear(self.NUM_HIDDENS, 1)
        self.lstmCell = nn.LSTMCell(self.NUM_LATENTS + self.NUM_ACTIONS, self.NUM_HIDDENS)

    def forward(self, inActions, inLSTMState, inLatents):
        """Predict the next latent distribution given the previous latent distribution
        
        Args:
            inActions: Action to predict for
            inLSTMState: Previous LSTM state
            inLatents: Previous latents (this is NOT a distribution)
        
        Returns:
            5-tuple: outMus, outSigmas, outMDLogProbs, outRewards, outTerminationProbs
            Returns means, sigmas and mixtures for the next latent, along with predictions
            for the next reward and whether the episode will terminate or not
        """
        inputs = torch.cat((inActions, inLatents), dim=1)
        outLSTMState, _ = self.lstmCell(inputs, inLSTMState) 

        BATCH_SIZE = self.inActions.shape[0]

        # Means of output latents
        outMus = self.muLayer(outLSTMState)
        outMus = outMus.view(BATCH_SIZE, self.NUM_GAUSSIANS, self.NUM_LATENTS)
        
        # Sigmas of output latents
        outSigmas = self.sigmaLayer(outLSTMState)
        outSigmas = outSigmas.view(BATCH_SIZE, self.NUM_GAUSSIANS, self.NUM_LATENTS)
        outSigmas = outSigmas.exp()

        # Log probabilities
        outMDLogProbs = self.MDLogProbLayer(outLSTMState)
        outMDLogProbs = outMDLogProbs.view(BATCH_SIZE, self.NUM_GAUSSIANS)
        outMDLogProbs = f.log_softmax(pi, dim=1)

        # Reward
        outRewards = self.rewardLayer(outLSTMState)

        # Episode termination probability
        outTerminationProbs = self.terminationProbLayer(outLSTMState)
        outTerminationProbs = torch.sigmoid(outTerminationProbs)

        return outMus, outSigmas, outMDLogProbs, outRewards, outTerminationProbs



