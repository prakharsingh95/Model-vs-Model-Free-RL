'''Extract frames from compressed numpy files that were created during
rollouts. Save these extracted frames by epoch (should call it
trajectory) for later processing by models.
'''
import cv2
import numpy as np
import os

from pathlib import Path
from tqdm import tqdm


cwd = Path(os.path.dirname(__file__))
record = cwd/'record'
images = cwd/'images'


# TODO: Accord this to dataset directory
def make_csv():
    filenames = []
    for epoch_idx in sorted(images.iterdir()):
        epoch = Path(epoch_idx)
        for filename in sorted(epoch.iterdir()):
            filenames.append(filename)
    with open('images.txt', 'w') as f:
        for filename in filenames:
            f.write(f'{filename}\n')


def extract_images():
    for epoch_idx, filename in enumerate(tqdm(list(record.iterdir()))):
        data = np.load(filename)['obs']
        epoch = images/f'epoch_{epoch_idx:04}'
        epoch.mkdir(parents=True, exist_ok=True)
        for frame_idx, image in enumerate(data):
            framename = (epoch/f'frame_{frame_idx:03}.png').as_posix()
            cv2.imwrite(framename, image)


def main():
#   extract_images()
    make_csv()


if __name__ == '__main__':
    main()
