import math

import numpy as np
import torch
from torch import nn


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class PedGraphConvolution(nn.Module):
    def __init__(self, cfg, input_ch, output_ch, v=17):
        super(PedGraphConvolution, self).__init__()
        self.cfg = cfg
        self.out_c = output_ch
        self.PA = nn.Parameter(torch.tensor(np.eye(v).astype(np.float32), requires_grad=True).to(cfg.device))
        self.alpha = nn.Parameter(torch.tensor(np.ones((v, v)).astype(np.float32), requires_grad=True).to(cfg.device))
        self.conv2d = nn.Conv2d(input_ch, output_ch, kernel_size=1)
        if input_ch != output_ch:
            self.pool = nn.Sequential(
                nn.Conv2d(input_ch, output_ch, kernel_size=1),
                nn.BatchNorm2d(output_ch)
            )
        else:
            self.pool = lambda x: x

        self.linear1 = nn.Linear(3, 3)
        self.linear2 = nn.Linear(3, 3)
        self.convT = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.linear3 = nn.Linear(3, 3)

        self.bn = nn.BatchNorm2d(output_ch)
        self.relu = nn.ReLU()
        self.reset_parameter()

    def reset_parameter(self):
        with torch.no_grad():
            self.PA /= torch.norm(self.PA, 2, dim=1, keepdim=True) + 1e-5
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        conv_branch_init(self.conv2d, 1)

    def forward(self, x):
        N, T, V, C = x.size()

        x_1 = self.linear1(x)
        x_2 = self.linear2(x)
        x_1 = x_1.unsqueeze(2)
        x_2 = x_2.unsqueeze(3)
        x_3 = (x_1 - x_2).permute(0, 1, 4, 2, 3)
        # N, T, C, V, V
        x_3 = x_3.view(N, T, C, V * V).permute(0, 2, 1, 3)
        x_3 = self.convT(x_3).squeeze(1).view(N, T, V, V)

        x_3 = torch.abs((x_3[:, 1:, :, :] - x_3[:, :-1, :, :]))
        x_3 = torch.mean(x_3.sum(dim=1), dim=0)

        x = x.view(N, C, T, V)


        # shared adj matrix
        z = self.conv2d(torch.matmul(x.view(N, C*T, V), self.PA + self.alpha * x_3).view(N, C, T, V))
        z = self.bn(z)
        z += self.pool(x)
        z = self.relu(z).reshape(N, T, V, self.out_c)

        return z


if __name__ == '__main__':
    x = torch.randn(64, 16, 17, 2)
    model = PedGraphConvolution(None, 2, 32)
    print(model(x).shape)