"""Definitions for various loss functions used in this project.
"""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def MixtureDensityLSTMLoss(trueLatents, predMu, predSigma, predMDLogProbs):
    """Computes the loss for a MixtureDensityLSTM model
    Loss per sample is defined as the negative of the log probability, which is then reduced across the batch
    
    Args:
        trueLatents: Actual latents observed (these are output by the VAE)
        predMu: Means of the predicted distribution for the true latents
        predSigma: Sigmas of the predicted distribution for the true latents
        predMDLogProbs: Mixture probabilities of the gaussians in the mixture model
    
    Returns:
        Loss, reduced across the batch
    """

    # Add a dimension such that it can be broadcasted to all gaussians in the mixture
    trueLatents = trueLatents.unsqueeze(-2)

    # Create a normal distribution using the predicted mu and sigma to compute log probability
    predDist = Normal(predMu, predSigma)
    logProbsByPredDist = predDist.log_prob(trueLatents)
    logProbsByPredDist = torch.sum(logProbsByPredDist, dim=-1)

    # Scale the probabilities in the distribution by the mixture probabilities
    logProbs = predMDLogProbs + logProbsByPredDist

    # Add the mixture probabilities to compute probability per batch
    logProbs = torch.logsumexp(logProbs, dim=-1)

    # Reduce loss across batch
    loss = - torch.mean(logProbs)

    return loss




