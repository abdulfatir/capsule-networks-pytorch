import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.funcs import squash


class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels, num_capsule_channels, dim_capsule, kernel_size, stride=1):
        super(PrimaryCapsule, self).__init__()
        self.dim_capsule = dim_capsule
        self.conv = nn.Conv2d(in_channels, num_capsule_channels * dim_capsule, kernel_size, stride)

    def forward(self, x):
        x = self.conv(x)
        x = x.view((x.size(0), -1, self.dim_capsule))
        x = squash(x)
        return x


class Capsule(nn.Module):
    def __init__(self, prev_num_capsules, prev_dim_capsule, num_capsules, dim_capsule, num_routing_iters=3):
        super(Capsule, self).__init__()
        self.prev_num_capsules = prev_num_capsules
        self.prev_dim_capsule = prev_dim_capsule
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.num_routing_iters = num_routing_iters
        self.W = torch.nn.Parameter(torch.randn(num_capsules, prev_num_capsules, prev_dim_capsule, dim_capsule))

    def forward(self, x):
        u_hat = torch.matmul(x[:, None, :, None, :], self.W[None, :, :, :, :])
        b = Variable(torch.zeros(*u_hat.size()))
        if torch.cuda.is_available():
            b = b.cuda()
        for i in range(1, self.num_routing_iters + 1):
            c = F.softmax(b, dim=2)
            v = squash((c * u_hat).sum(dim=2, keepdim=True))
            if i is not self.num_routing_iters:
                db = (v * u_hat).sum(dim=-1, keepdim=True)
                b = b + db
        return v.squeeze()

    def __repr__(self):
        s = '{}(prev_num_capsules={}, prev_dim_capsule={}, num_capsules={}, dim_capsule={}, num_routing_iters={})'
        return s.format(self.__class__.__name__, self.prev_num_capsules, self.prev_dim_capsule, self.num_capsules, self.dim_capsule, self.num_routing_iters)