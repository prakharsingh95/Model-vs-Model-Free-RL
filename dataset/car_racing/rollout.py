#!/usr/bin/env python3
'''Generate a rollout using a random policy. This is useful for getting data to
trian the VAE.
'''

import argparse
import cv2
import json
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import settings

from PIL import Image
from gym.envs.box2d.car_racing import CarRacing
from gym.spaces.box import Box
from pathlib import Path
from skimage.transform import resize
from torchvision import transforms

from utils.misc import RolloutGenerator


class CarRacingWrapper(CarRacing):
    def __init__(self):
        super(CarRacingWrapper, self).__init__()
        self.width = 64
        self.height = 64
        self.observation_space = Box(low=0, high=255,
                                     shape=(self.width, self.height, 3))

    def step(self, action):
        state, reward, done, _ = super(CarRacingWrapper, self).step(action)
        return self.process_frame(state), reward, done, _

    def process_frame(self, frame):
        # Cut off last 10 pixels since these are UI for human
        frame = frame[0:84, :, :]
        # Reshape to 64x64x3
        frame = Image.fromarray(frame)
        frame = frame.resize((self.width, self.height),
                             resample=Image.BILINEAR)
        frame = np.array(frame)
        return frame


def parse_args():
    cwd = Path(os.path.dirname(__file__))
    rollout = cwd/'rollout'
    parser = argparse.ArgumentParser(description='Rollout generator')
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='Number of rollouts to run')
    parser.add_argument('--savedir', type=str, default=f'{cwd}/rollout',
                        help='Location to save rollouts')
    parser.add_argument('--policy', type=str, default='random',
                        help='random, brown, controller')
    args = parser.parse_args()
    return args


def brownian_sample(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    # Taken unmodified from ctallec implementation
    actions = [action_space.sample()]
    for _ in range(1, seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + np.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions


def rollout(env, savedir, policy):
    images = Path('images')
    max_frames = 1000
    num_seen_frames = 0
    done = False
    metadata = []

    if policy == 'random':
        actions = [env.action_space.sample() for _ in range(max_frames+1)]
    elif policy == 'brown':
        actions = brownian_sample(env.action_space, max_frames+1, dt=1/50)

    hidden = [
        torch.zeros(1, settings.mdrnn_hidden_dim).to(settings.device)
        for _ in range(2)]
    rg = RolloutGenerator(mdir=Path(os.environ['top']), device=settings.device,
                          time_limit=1000)

    obs = env.reset()

    while not done:
        if policy in ['random', 'brown']:
            action = actions[num_seen_frames]
        elif policy == 'controller':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])
            obs = transform(obs).unsqueeze(0).to(settings.device)
            action, hidden = rg.get_action_and_transition(obs, hidden)
        env.render('rgb_array')  # Look into why this call is necessary.

        obs, reward, done, _ = env.step(action)

        if num_seen_frames == max_frames:
            done = True
            reward = -100.0

        if num_seen_frames > 0:
            cv2.imwrite(f'{savedir}/frame_{num_seen_frames:04}.png', obs)
            metadata.append(
                dict(idx=num_seen_frames,
                     action=action.tolist(),
                     reward=reward, done=done))
        num_seen_frames += 1

    with open(f'{savedir}/metadata.json', 'w') as f:
        content = json.dumps(metadata, indent=4)
        f.write(content)


def main():
    args = parse_args()
    env = CarRacingWrapper()
    for i in range(args.num_rollouts):
        savedir = Path(args.savedir)/f'trajectory_{np.random.randint(2**31-1)}'
        savedir.mkdir(exist_ok=True, parents=True)
        rollout(env, savedir, policy=args.policy)


if __name__ == '__main__':
    main()
