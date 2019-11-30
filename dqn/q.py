import torch
import torch.nn as nn
import torch.nn.functional as F

class Q(nn.Module):
    def __init__(self, channels, height, width, num_actions):
        super(Q, self).__init__()

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.hidden = nn.Linear((height//16) * (width//16) * 32, 256)

        self.final = nn.Linear(256, num_actions)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(x.shape[0], -1)

        x = F.relu(self.hidden(x))

        x = self.final(x)
 
        return x