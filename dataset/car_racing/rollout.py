#!/usr/bin/env python3

import argparse
import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import json

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
    savedir = cwd/'rollout'
    parser.add_argument('--savedir', type=str, default=f'{savedir}',
                        help='Location to save rollouts')
    args = parser.parse_args()
    return args


def rollout(env, savedir):
    images = Path('images')
    random_val = np.random.randint(0, 2**31-1)
    max_frames = 1000
    env.reset()
    num_seen_frames = 0
    done = False
    metadata = []

    state = None
    while not done:
        # TODO: Get action from RNN
        action = env.action_space.sample()
        env.render('rgb_array')  # Look into why this call is necessary.
        next_state, reward, done, _ = env.step(action)

        if num_seen_frames >= max_frames:
            done = True
            reward = -100.0

        if state is not None:
            cv2.imwrite(f'{savedir}/frame_{num_seen_frames:04}.png', state)
            metadata.append(
                dict(idx=num_seen_frames,
                     action=action.tolist(),
                     reward=reward, done=done))
            num_seen_frames += 1
        state = next_state


    with open(f'{savedir}/metadata.json', 'w') as f:
        content = json.dumps(metadata, indent=4)
        f.write(content)


def main():
    args = parse_args()
    env = CarRacingWrapper()
    for i in range(args.num_rollouts):
        savedir = Path(args.savedir)/f'trajectory_{np.random.randint(2**31-1)}'
        savedir.mkdir(exist_ok=True, parents=True)
        rollout(env, savedir)


if __name__ == '__main__':
    main()
