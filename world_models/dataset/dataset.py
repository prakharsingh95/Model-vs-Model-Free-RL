'''Return a dataset that generates images from a csv file.'''

import numpy as np

import matplotlib.pyplot as plt
import os
import torch
import json

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


class CSVSequenceDataset(Dataset):

    """Dataset for generating sequences of states, action, rewards, dones
    and next state tuples to be used to train the MDRNN
    """

    def __init__(self, dirs_file: str, seq_len: int):
        """Constructor, sets up the list of feasible first files.
        
        Args:
            dirs_file (str): List of trajectory directories to look into
            seq_len (int): Length of sequence to return
        
        Raises:
            Exception: If sequence length requested is bigger than trajectory
            length, an exception is raised.
        """
        self.cwd = Path(os.path.dirname(__file__))
        dirs = open(dirs_file).read().splitlines()
        self._first_files = []
        for _dir in dirs:
            _dir_path = self.cwd/_dir
            files_in_dir = [file.as_posix()
                            for file in sorted(_dir_path.iterdir())]

            # To get next state as well, need seq_len+1 samples
            if len(files_in_dir) <= seq_len-1:
                raise Exception("Can't sample this big a sequence!")
            self._first_files += files_in_dir[:-(seq_len+1)]
        self.seq_len = seq_len
        self.transform = ToTensor()

    def __len__(self):
        """Length of possible sequences is equal to the number of feasible
        starting states. This can possibly be optimized to read images
        in chunks for faster performance
        """
        return len(self._first_files)

    def __getitem__(self, idx):
        """Returns an item at index idx. File path stored in self._first_files
        serves as the starting point for the sequence
        
        Args:
            idx (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        first_file_path = Path(self._first_files[idx])
        folder = first_file_path.parent
        in_data_file_num = int(first_file_path.as_posix()[-8:-4])

        with open(folder/'metadata.json', 'r') as f:
            content = f.read()
            metadata = json.loads(content)
        # TODO: generalize this using regex

        states, actions, rewards, dones = [], [], [], []
        for i in range(in_data_file_num, in_data_file_num + self.seq_len + 1):
            file_path = folder/f'frame_{i:04}.png'
            image = plt.imread(file_path.as_posix())
            image = self.transform(image)
            states.append(image)
            actions.append(metadata[i]['action'])
            rewards.append(metadata[i]['reward'])
            dones.append(metadata[i]['done'])

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        cur_states = states[:-1]
        actions = actions[:-1]
        rewards = rewards[:-1].reshape(-1, 1)
        dones = dones[:-1].reshape(-1, 1)
        next_states = states[1:]

        return cur_states, actions, rewards, dones, next_states
