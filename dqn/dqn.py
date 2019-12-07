
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .q import Q
from utils import MultiStepEnv, ReplayBuffer

import settings

from pympler import muppy, summary

class DQN(object):

    def __init__(self, 
        multi_step_env: MultiStepEnv = None,
        gamma: float = None,
        eps_max: float = None,
        eps_min: float = None,
        eps_decay_steps: int = None,
        replay_min_size: int = None,
        replay_max_size: int = None,
        target_update_freq: int = None,
        steps_per_update: int = None,
        train_batch_size: int = None,
        enable_rgb: bool = None,
        model_save_file: str = None,
        optim_l2_reg_coeff: float = None,
        optim_lr: float = None,
        eval_freq: int = None
    ):

        self.env = multi_step_env
        self.gamma = gamma
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay_steps = eps_decay_steps
        self.replay_min_size = replay_min_size
        self.target_update_freq = target_update_freq
        self.train_batch_size = train_batch_size
        self.steps_per_update = steps_per_update
        self.model_save_file = model_save_file
        self.optim_lr = optim_lr
        self.optim_l2_reg_coeff = optim_l2_reg_coeff
        self.eval_freq = eval_freq

        self.replay_memory = ReplayBuffer(capacity = replay_max_size)
        self.n_steps = 0

        if enable_rgb:
            self.q_train = Q(self.env.frame_stack_size*3, self.env.height, self.env.width, self.env.num_actions).to(settings.device)
            self.q_target = Q(self.env.frame_stack_size*3, self.env.height, self.env.width, self.env.num_actions).to(settings.device)
        else:
            self.q_train = Q(self.env.frame_stack_size, self.env.height, self.env.width, self.env.num_actions).to(settings.device)
            self.q_target = Q(self.env.frame_stack_size, self.env.height, self.env.width, self.env.num_actions).to(settings.device)

        self.optimizer = Adam(self.q_train.parameters(), eps=1e-7, lr=self.optim_lr, weight_decay=self.optim_l2_reg_coeff)
        # self.mse_loss = nn.MSELoss()
        assert(self.q_train.state_dict().keys() == self.q_target.state_dict().keys())

    def _update_step_counter(self):
        self.n_steps += 1

    def copyAtoB(self, A, B, tau=None):
        for paramA, paramB in zip(A.parameters(), B.parameters()):
            paramB.data.copy_(paramA.data)

    def _update_q_target(self):
        if (self.n_steps % self.target_update_freq) == 0:
            self.copyAtoB(self.q_train, self.q_target)
            

    @staticmethod
    def normalize(obs):
        return (obs/255.0 * 2 - 1)

    def _update_q_train(self):
        if self.replay_memory.size >= self.replay_min_size and (self.n_steps % self.steps_per_update) == 0:
            self.q_train.train()

            states, action_idxs, rewards, next_states, dones = self.replay_memory.sample(self.train_batch_size)

            states = torch.FloatTensor(states).to(settings.device).permute(0,3,1,2)
            action_idxs = torch.LongTensor(action_idxs).to(settings.device)
            rewards = torch.FloatTensor(rewards).to(settings.device)
            next_states = torch.FloatTensor(next_states).to(settings.device).permute(0,3,1,2)
            dones = torch.FloatTensor(dones).to(settings.device)

            states = self.normalize(states)
            next_states = self.normalize(next_states)


            q_cur = torch.gather(self.q_train(states), -1, action_idxs.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                q_next = self.q_target(next_states)
                v_next, _ = torch.max(q_next, dim=-1)
                targets = rewards + self.gamma * v_next * (1-dones)
                targets = targets.detach()

            loss = 0.5 * F.mse_loss(q_cur, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.detach().cpu().item()
        return None

    def _get_eps_greedy_action(self, obs, eps=0):
        if np.random.rand() < eps:
            return self.env.random_action()
        else:
            self.q_train.eval()
            obs = self.normalize(obs)
            obs = torch.FloatTensor(obs).to(settings.device)
            obs = obs.unsqueeze(0).permute(0,3,1,2)
            q = self.q_train(obs).squeeze(0)
            return torch.argmax(q).detach().cpu().item()


    def _get_epsilon(self):
        eps = self.eps_min + max(0, (self.eps_decay_steps - self.n_steps)/self.eps_decay_steps) * (self.eps_max - self.eps_min)
        return eps

    def save_state(self, file):
        checkpoint = {}
        checkpoint['q_train'] = self.q_train.state_dict()
        checkpoint['q_target'] = self.q_target.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        torch.save(checkpoint, file)

    def load_state(self, file):
        checkpoint = torch.load(file)
        self.q_train.load_state_dict(checkpoint['q_train'])
        self.q_target.load_state_dict(checkpoint['q_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self, num_episodes):

        all_rewards = []

        for n_episode in range(num_episodes):
    
            # Log test episode
            if n_episode % self.eval_freq == 0:
                self.eval(1)

            # Reset the environment
            obs = self.env.reset()

            # Play an episode
            total_reward = 0.0
            total_loss = 0.0
            n = 0
            done = False
            
            while not done:
                # Epsilon greedy action selection
                action_idx = self._get_eps_greedy_action(obs, eps=self._get_epsilon())

                # Take a step
                next_obs, reward, done, _ = self.env.step(action_idx)

                # nobs, nnext = self.normalize(obs), self.normalize(next_obs)
                # print(np.mean(np.abs(nobs[:,:,0], nnext[:,:,0])))
                # print(self.normalize(obs))

                # Update replay buffer
                self.replay_memory.insert(obs, action_idx, reward, next_obs, done)

                # Update networks
                self._update_q_target()
                loss = self._update_q_train()

                # Bookkeeping
                total_reward += reward
                self._update_step_counter()
                if loss is not None:
                    total_loss += loss
                
                n += 1

                obs = next_obs

            # Save weights
            self.save_state(self.model_save_file)

            all_rewards.append(total_reward)
            if(len(all_rewards) > 100):
                all_rewards = list(np.array(all_rewards)[-100:])

            last_100_avg_rwd = np.sum(all_rewards) / len(all_rewards)

            avg_loss = total_loss * 4.0/n
            print('[TRAIN] n_episode: {}/{}, steps: {}, total_steps: {}, episode_reward: {:.03f}, 100_avg_rwd: {:.03f}, avg_loss: {:.03f}, eps: {:.03f}, replay_size: {}'\
                .format(n_episode+1, num_episodes, n, self.n_steps, total_reward, last_100_avg_rwd, avg_loss, self._get_epsilon(), self.replay_memory.size))

    def eval(self, num_episodes):
        all_rewards = []

        for n_episode in range(num_episodes):
    
            # Reset the environment
            obs = self.env.reset(video_log_file=f'dqn_test_log_episode_{n_episode}')

            # Play an episode
            total_reward = 0.0
            total_loss = 0.0
            n = 0
            done = False
            
            while not done:
                # Greedy action selection
                action_idx = self._get_eps_greedy_action(obs, eps=0.0)

                # Take a step
                next_obs, reward, done, _ = self.env.step(action_idx)

                # Bookkeeping
                total_reward += reward                
                n += 1
                obs = next_obs


            all_rewards.append(total_reward)

            avg_rwd = np.sum(all_rewards) / len(all_rewards)

            print('[EVAL] n_episode: {}/{}, steps: {}, episode_reward: {:.03f}, avg_rwd: {:.03f}'\
                .format(n_episode+1, num_episodes, n, total_reward, avg_rwd))






