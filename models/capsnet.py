import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from layers import PrimaryCapsule, Capsule


class CapsNet(torch.nn.Module):
    def __init__(self, num_primary_capsules, input_shape=[1, 28, 28], n_classes=10):
        super(CapsNet, self).__init__()

        self.CUDA = torch.cuda.is_available()
        self.input_shape = input_shape
        self.conv1 = torch.nn.Conv2d(input_shape[0], 256, 9)
        self.primary_capsule = PrimaryCapsule(256, 32, 8, 9, 2)
        self.digit_capsule = Capsule(num_primary_capsules, 8, n_classes, 16)
        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, int(np.prod(input_shape))),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsule(x)
        v = self.digit_capsule(x)
        class_probs = F.softmax(torch.sqrt((v ** 2).sum(dim=-1)), dim=-1)
        if y is not None:
            indices = y
        else:
            _, indices = torch.max(class_probs, dim=-1)
        mask = Variable(torch.eye(10))
        if self.CUDA:
            mask = mask.cuda()
        mask = mask.index_select(dim=0, index=indices)
        recons = self.decoder((v * mask[:, :, None]).view(batch_size, -1))
        shape = [-1] + self.input_shape
        recons = recons.view(shape)
        return class_probs, recons
