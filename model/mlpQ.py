import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import operator as op
import torch
from recovery.qat.qat import Linear_Q, Conv2d_Q

class MLPQ(nn.Module, ):
        def __init__(self, in_channels):
            super(MLPQ, self).__init__()
            self.nb_classes = 10
            self.in_channels = in_channels
            self.img_rows = 28
            self.img_cols = 28
            self.fc1 = Linear_Q(in_features=self.in_channels*self.img_rows*self.img_cols, out_features=300)
            self.fc2 = Linear_Q(in_features=300, out_features=100)
            self.fc3 = Linear_Q(in_features=100, out_features=self.nb_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.view(-1, self.in_channels*self.img_rows*self.img_cols)
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.fc3(out)
            return F.softmax(out,dim=1)


