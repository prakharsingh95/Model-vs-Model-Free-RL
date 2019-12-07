'''Global settings.'''

import torch

# TODO: Capitalize all constants across the project

batch_size = 32
cuda = True
device = torch.device('cuda')
log_interval = 20
num_epochs = 30
seed = 2718
train_test_split = 0.8
VIDEO_LOG_DIR = 'videos'


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

# DQN parameters
DQN_TRAIN_BATCH_SIZE = 64
DQN_FRAME_STACK_SIZE = 4
DQN_STEPS_PER_UPDATE = 4
DQN_FRAME_SKIPS = 2
DQN_GAMMA = 0.99
DQN_EPS_MAX = 1.0
DQN_EPS_MIN = 0.1
DQN_EPS_DECAY_STEPS = 100000
DQN_REPLAY_MIN_SIZE = 1000
DQN_REPLAY_MAX_SIZE = 40000
DQN_TARGET_UPDATE_FREQ = 1000
DQN_OPTIM_LR = 4e-4
DQN_OPTIM_L2_REG_COEFF = 1e-6
DQN_ENABLE_RGB = True
DQN_EVAL_FREQ = 50
DQN_RENDER_MODE = 'direct'
DQN_WEIGHTS_SAVE_FILE = 'dqn.pt'
