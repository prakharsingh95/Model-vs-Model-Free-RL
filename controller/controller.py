'''Controller implementation.'''

from torch import nn


class Controller(nn.Module):
    def __init__(self, input_size):
        super(Controller, self).__init__()
        action_size = 3
        # Does nn.Linear include bias?
        self.fc = nn.Linear(input_size, action_size)

    def forward(self, *x):
        # What does dim=1 do?
        cat = torch.cat(x, dim=1)
        return self.fc(cat)
