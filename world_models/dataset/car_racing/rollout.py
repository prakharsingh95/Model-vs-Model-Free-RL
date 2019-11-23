#!/usr/bin/env python3

import argparse
import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy

from PIL import Image
from gym.envs.box2d.car_racing import CarRacing
from gym.spaces.box import Box
from pathlib import Path
from skimage.transform import resize


class CarRacingWrapper(CarRacing):
    def __init__(self):
        super(CarRacingWrapper, self).__init__()
        self.width = 64
        self.height = 64
        self.observation_space = Box(low=0, high=255,
                                     shape=(self.width, self.height, 3))

    def step(self, action):
        state, _, done, _ = super(CarRacingWrapper, self).step(action)
        return self.process_frame(state), _, done, _

    def process_frame(self, frame):
        # Try to switch this to cv2 resize
        # Reshape to 64x64x3
        # frame = scipy.misc.imresize(frame, (self.width, self.height))
        frame = Image.fromarray(frame)
        frame = frame.resize((self.width, self.height), resample=Image.BILINEAR)
        frame = np.array(frame)

        # Cut off last 10 pixels since these are UI for human
        frame = frame[0:84,:,:].astype(np.float)/255.0

        frame = ((1.0 - frame) * 255).round().astype(np.uint8)
        return frame


def parse_args():
    cwd = Path(os.path.dirname(__file__))
    rollout = cwd/'rollout'
    parser = argparse.ArgumentParser(description='Rollout generator')
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='Number of rollouts to run')
    savedir = cwd/'rollout'
    parser.add_argument('--savedir', type=str, default=f'{savedir}',
                        help='Location to save rollouts')
    args = parser.parse_args()
    return args


def rollout(env, savedir):
    images = Path('images')
    random_val = np.random.randint(0, 2**31-1)
    max_frames = 200
    env.reset()
    num_seen_frames = 0
    done = False
    while num_seen_frames < max_frames and not done:
        # TODO: Get action from RNN
        action = env.action_space.sample()
        env.render('rgb_array')  # Look into why this call is necessary.
        state, _, done, _ = env.step(action)
        cv2.imwrite(f'{savedir}/frame_{num_seen_frames:04}.png', state)
        num_seen_frames += 1


def main():
    args = parse_args()
    env = CarRacingWrapper()
    for i in range(args.num_rollouts):
        savedir = Path(args.savedir)/f'trajectory_{np.random.randint(2**31-1)}'
        savedir.mkdir(exist_ok=True, parents=True)
        rollout(env, savedir)


if __name__ == '__main__':
    main()
