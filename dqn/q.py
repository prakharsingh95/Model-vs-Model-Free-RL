import torch
import torch.nn as nn
import torch.nn.functional as F

class Q(nn.Module):
    def __init__(self, channels, height, width, num_actions):
        super(Q, self).__init__()

        self.conv1 = nn.Conv2d(channels, 8, kernel_size=7, stride=3, padding=3)
        self.mp1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.mp2 = nn.MaxPool2d(2, stride=2)
        self.hidden = nn.Linear(1024, 256)
        self.final = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.hidden(x))
        x = self.final(x)
        return x