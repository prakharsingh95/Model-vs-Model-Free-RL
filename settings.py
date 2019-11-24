'''Global settings.'''

import torch


batch_size = 32
cuda = True
device = torch.device('cuda')
log_interval = 20
num_epochs = 30
seed = 2718
train_test_split = 0.8


# Misc parameters
action_space_size = 3
reduced_image_channels = 3
reduced_image_height = 64
reduced_image_width = 64

# VAE Parameters
vae_latent_dim = 32

# MDRNN Parameters
num_epochs_mdrnn = 30
mdrnn_batch_size = 16
mdrnn_save_freq = 100

mdrnn_hidden_dim = 256
mdrnn_num_gaussians = 5
mdrnn_train_seq_len = 32
mdrnn_train_lr = 1e-3
