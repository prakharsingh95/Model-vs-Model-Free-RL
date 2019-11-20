'''Return a dataset that generates images from a csv file rooted with
root_dir.
'''

import matplotlib.pyplot as plt
import torch

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
    def __init__(self, csv_file, root_dir):
        self.filenames = open(csv_file).read().splitlines()
        self.root_dir = root_dir
        self.transform = ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = (self.root_dir/self.filenames[idx]).as_posix()
        image = plt.imread(filename)
        return self.transform(image)
