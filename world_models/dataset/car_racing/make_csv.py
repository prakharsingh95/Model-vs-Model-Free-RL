#!/usr/bin/env python3

import cv2
import numpy as np
import os

from pathlib import Path
from tqdm import tqdm

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
    split = int(0.8*len(files))
    train_files = files[:split]
    test_files = files[split:]
    with open('train_images.txt', 'w') as f:
        f.write('\n'.join(train_files))
    with open('test_images.txt', 'w') as f:
        f.write('\n'.join(test_files))


def main():
    make_csv()


if __name__ == '__main__':
    main()
