import os
import argparse
from pathlib import Path

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader

import settings
import dataset
from vae import VAE
from mdrnn import MixtureDensityLSTM
from losses import MixtureDensityLSTMLoss

cwd = Path(os.path.dirname(__file__))
parser = argparse.ArgumentParser(description='MDRNN Trainer')
parser.add_argument('--vae_savefile', type=str, default=f'{cwd}/vae.pt')
parser.add_argument('--mdrnn_savefile', type=str,
                    default=f'{cwd}/mdrnn.pt')
parser.add_argument('--include_reward', action='store_true')
args = parser.parse_args()

vae = VAE(input_size=(settings.reduced_image_channels,
                          settings.reduced_image_height,
                          settings.reduced_image_width),
              latent_dim=settings.vae_latent_dim)\
        .to(settings.device)
vae.load_state_dict(torch.load(f'{args.vae_savefile}'))
vae.eval()

mdrnn = MixtureDensityLSTM(settings.vae_latent_dim,
                           settings.action_space_size,
                           settings.mdrnn_hidden_dim,
                           settings.mdrnn_num_gaussians).to(settings.device)

if Path(args.mdrnn_savefile).exists():
    print('Starting MDRNN training from checkpoint...')
    mdrnn.load_state_dict(torch.load(f'{args.mdrnn_savefile}'))

optimizer = Adam(mdrnn.parameters(), lr=settings.mdrnn_train_lr)

def get_loss(batch):
    cur_states, actions, rewards, dones, next_states = batch

    cur_states = cur_states.to(settings.device)
    actions = actions.to(settings.device)
    rewards = rewards.to(settings.device)
    dones = dones.to(settings.device)
    next_states = next_states.to(settings.device)

    BATCH_SIZE, SEQ_LEN, CHANNELS, HEIGHT, WIDTH = cur_states.shape

    _, cur_z, _, _ = vae(cur_states.view(-1, CHANNELS, HEIGHT, WIDTH))
    _, next_z, _, _ = vae(next_states.view(-1, CHANNELS, HEIGHT, WIDTH))

    cur_z = cur_z.view(BATCH_SIZE, SEQ_LEN, -1).detach()
    next_z = next_z.view(BATCH_SIZE, SEQ_LEN, -1).detach()

    pred_mus, pred_sigmas, pred_md_logprobs, pred_rewards, pred_termination_probs \
        = mdrnn(actions, cur_z)

    mixture_loss = MixtureDensityLSTMLoss(next_z, pred_mus, pred_sigmas, pred_md_logprobs)

    terminal_loss = F.binary_cross_entropy(pred_termination_probs, dones)

    reward_loss = F.mse_loss(pred_rewards, rewards)

    if args.include_reward:
        loss = (mixture_loss + terminal_loss + reward_loss) / (settings.vae_latent_dim + 2)
    else:
        loss = (mixture_loss + terminal_loss) / (settings.vae_latent_dim + 1)

    return loss

def train_on_data(epoch, train_loader):
    mdrnn.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        BATCH_SIZE = batch[0].shape[0]

        loss = get_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / BATCH_SIZE

        if (batch_idx % settings.log_interval) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item() / BATCH_SIZE))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader)))

def test_on_data(epoch, test_loader):
    mdrnn.eval()
    test_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            BATCH_SIZE = batch[0].shape[0]
            loss = get_loss(batch)
            test_loss += loss.item() / BATCH_SIZE
    test_loss /= len(test_loader)
    print('====> Test set loss: {:.4f}'.format(test_loss))


train_csv = os.path.join('car_racing', 'train_seq_dirs.txt')
test_csv = os.path.join('car_racing', 'test_seq_dirs.txt')
train_dataset = dataset.CSVSequenceDataset(train_csv, settings.mdrnn_train_seq_len)
test_dataset = dataset.CSVSequenceDataset(test_csv, settings.mdrnn_train_seq_len)
kwargs = {'num_workers': 1, 'pin_memory': True} if settings.cuda else {}
train_loader = DataLoader(train_dataset, batch_size=settings.mdrnn_batch_size,
                          shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=settings.mdrnn_batch_size,
                         shuffle=True, **kwargs)

for epoch in range(settings.num_epochs_mdrnn):
    train_on_data(epoch, train_loader)
    test_on_data(epoch, test_loader)
torch.save(mdrnn.state_dict(), f'{args.mdrnn_savefile}')
