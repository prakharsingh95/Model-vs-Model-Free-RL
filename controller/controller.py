'''Controller implementation.'''

import torch

from torch import nn


class Controller(nn.Module):
    def __init__(self, input_size, output_size):
        super(Controller, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, *x):
        cat = torch.cat(x, dim=1)
        return self.fc(cat)
