'''Return a dataset that generates images from a csv file.'''

import matplotlib.pyplot as plt
import os
import torch

from pathlib import Path
from torch.utils.data import Dataset


class ToTensor():
    def __call__(self, image):
        '''Make color axis first dimension to accord with pytorch. This
        assumes the image has the color axis as the third dimension, as
        it will through plt.imread.
        '''
        image = image.transpose((2, 0, 1))
        return image


class CSVDataset(Dataset):
    '''This assumes that the csv contains filenames relative to the
    directory this file is located.
    '''
    def __init__(self, csv_file):
        self.cwd = Path(os.path.dirname(__file__))
        csv_file = self.cwd/csv_file
        self.filenames = open(csv_file).read().splitlines()
        self.transform = ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = (self.cwd/self.filenames[idx]).as_posix()
        image = plt.imread(filename)
        return self.transform(image)
