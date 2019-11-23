#!/usr/bin/env python3

import cv2
import numpy as np
import os

from pathlib import Path
from tqdm import tqdm

import settings

cwd = Path(os.path.dirname(__file__))
rollout = cwd/'rollout'

def make_csv():
    files = []
    for i in sorted(rollout.iterdir()):
        trajectory = Path(i)
        for file in sorted(trajectory.iterdir()):
            # TODO: Abstract this
            files.append(('car_racing'/file).as_posix())
    np.random.shuffle(files)
    split = int(settings.train_test_split*len(files))
    train_files = files[:split]
    test_files = files[split:]
    with open('train_images.txt', 'w') as f:
        f.write('\n'.join(train_files))
    with open('test_images.txt', 'w') as f:
        f.write('\n'.join(test_files))

#TODO: refactor redundant parts in these two calls
def make_seq_csv():
    dirs = []
    for _dir in sorted(rollout.iterdir()):
        dirs.append(('car_racing'/_dir).as_posix())

    np.random.shuffle(dirs)
    split = int(settings.train_test_split*len(dirs))

    train_dirs = dirs[:split]
    test_dirs = dirs[split:]

    with open('train_seq_dirs.txt', 'w') as f:
        f.write('\n'.join(train_dirs))
    with open('test_seq_dirs.txt', 'w') as f:
        f.write('\n'.join(test_dirs))

def main():
    make_csv()
    make_seq_csv()


if __name__ == '__main__':
    main()
