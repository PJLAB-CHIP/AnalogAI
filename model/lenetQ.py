'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from QModule import Linear_Q, Conv2d_Q

class LeNetQ(nn.Module):
    def __init__(self):
        super(LeNetQ, self).__init__()
        self.conv1 = Conv2d_Q(3, 6, 5, first_layer = 1)
        self.conv2 = Conv2d_Q(6, 16, 5)
        self.fc1   = Linear_Q(16*5*5, 120)
        self.fc2   = Linear_Q(120, 84)
        self.fc3   = Linear_Q(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
