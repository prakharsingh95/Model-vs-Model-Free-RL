import torch
import unittest

import settings
from mdrnn import MixtureDensityLSTM, MixtureDensityLSTMCell


class TestMDRNN(unittest.TestCase):

    def setUp(self):
        self.mdrnn = MixtureDensityLSTM(settings.vae_latent_dim,
                                        settings.action_space_size,
                                        settings.mdrnn_hidden_dim,
                                        settings.mdrnn_num_gaussians) \
            .to(settings.device)

        self.mdrnn_cell = MixtureDensityLSTMCell(settings.vae_latent_dim,
                                                 settings.action_space_size,
                                                 settings.mdrnn_hidden_dim,
                                                 settings.mdrnn_num_gaussians)\
            .to(settings.device)
        self.mdrnn_cell.load_weights(self.mdrnn)

    def test_mdrnn(self):

        BATCH_SIZE = 32
        SEQ_LEN = 32
        in_actions = torch.normal(
            0, 1, (BATCH_SIZE, SEQ_LEN, settings.action_space_size))\
            .to(settings.device)
        in_latents = torch.normal(
            0, 1, (BATCH_SIZE, SEQ_LEN, settings.vae_latent_dim))\
            .to(settings.device)

        pred_mus, pred_sigmas, pred_md_logprobs, pred_rewards, \
            pred_termination_probs, pred_sampled_z \
            = self.mdrnn(in_actions, in_latents)

        self.assertEqual(pred_mus.shape, (BATCH_SIZE, SEQ_LEN,
                                          settings.mdrnn_num_gaussians,
                                          settings.vae_latent_dim))

    def test_mdrnn_cell(self, eps=1e-4):
        BATCH_SIZE = 1
        SEQ_LEN = 1000
        in_actions = torch.normal(
            -0.5, 1, (BATCH_SIZE, SEQ_LEN, settings.action_space_size))\
            .to(settings.device)
        in_latents = torch.normal(
            1, 1, (BATCH_SIZE, SEQ_LEN, settings.vae_latent_dim))\
            .to(settings.device)

        lstm_mus, lstm_sigmas, lstm_md_logprobs, lstm_rewards,\
            lstm_termination_probs, lstm_sampled_z \
            = self.mdrnn(in_actions, in_latents)

        in_hidden = (torch.zeros(BATCH_SIZE, settings.mdrnn_hidden_dim).to(
            settings.device),
            torch.zeros(BATCH_SIZE, settings.mdrnn_hidden_dim)
            .to(settings.device))

        for t in range(SEQ_LEN):
            cell_mus, cell_sigmas, cell_md_logprobs, cell_rewards, \
                cell_termination_probs, cell_hiddens \
                = self.mdrnn_cell(in_actions[:, t, :],
                                  in_hidden,
                                  in_latents[:, t, :])
            self.assertTrue(
                torch.sum(torch.abs(
                    cell_mus - lstm_mus[:, t, :, :])).item() <= eps)
            self.assertTrue(
                torch.sum(torch.abs(
                    cell_sigmas - lstm_sigmas[:, t, :, :])).item() <= eps)
            self.assertTrue(torch.sum(torch.abs(
                cell_md_logprobs - lstm_md_logprobs[:, t, :])).item() <= eps)
            self.assertTrue(
                torch.sum(torch.abs(
                    cell_rewards - lstm_rewards[:, t, :])).item() <= eps)
            self.assertTrue(torch.sum(torch.abs(
                cell_termination_probs - lstm_termination_probs[:, t, :]))
                .item() <= eps)

            in_hidden = cell_hiddens


if __name__ == '__main__':
    unittest.main()
