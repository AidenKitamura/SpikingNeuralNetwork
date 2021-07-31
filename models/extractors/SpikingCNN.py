import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

import numpy as np
import math as m
from SpikingLayer import SpikingLayer, SpikingVextLayer, poisson_spikes

class SpikingConvNet(nn.Module):
    
    def __init__(self, Nin, Nout, Nsp, t1, t2, beta=5, scale=1):
        super(SpikingConvNet, self).__init__()
        self.Nsp = Nsp
        self.Nout = Nout
        self.Nin = Nin
        self.Nhid1 = Nin
        self.Nhid2 = 600
        self.scale = scale
        # self.conv1 = nn.Conv2d(1, 4, (5,5), stride=2, padding=2)
        # Need to change this coefficient
        self.l1 = nn.Linear(self.Nhid2, self.Nout, bias=None)
        # self.conv2 = nn.Conv2d(4, 6, (5,5), stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 2,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv2 = nn.Conv2d(in_channels = 2, out_channels = 4,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv3 = nn.Conv2d(in_channels = 4, out_channels = 8,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv4 = nn.Conv2d(in_channels = 8, out_channels = 16,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv5 = nn.Conv2d(in_channels = 16, out_channels = 32,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv6 = nn.Conv2d(in_channels = 32, out_channels = 64,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))

        self.sl1 = SpikingLayer(t1, beta=beta)
        self.sl2 = SpikingLayer(t1, beta=beta)
        self.sl3 = SpikingLayer(t1, beta=beta)
        self.sl4 = SpikingLayer(t1, beta=beta)
        self.sl5 = SpikingLayer(t1, beta=beta)
        self.sl6 = SpikingLayer(t1, beta=beta)
        self.sl7 = SpikingLayer(t2, beta=beta)

    def forward(self, x, device):
        s1 = torch.zeros(x.shape[0], 2, 28, 28).to(device)
        v1 = torch.zeros(x.shape[0], 2, 28, 28).to(device)
        s2 = torch.zeros(x.shape[0], 4, 28, 28).to(device)
        v2 = torch.zeros(x.shape[0], 4, 28, 28).to(device)
        s3 = torch.zeros(x.shape[0], 8, 14, 14).to(device)
        v3 = torch.zeros(x.shape[0], 8, 14, 14).to(device)
        s4 = torch.zeros(x.shape[0], 16, 14, 14).to(device)
        v4 = torch.zeros(x.shape[0], 16, 14, 14).to(device)
        s5 = torch.zeros(x.shape[0], 32, 7, 7).to(device)
        v5 = torch.zeros(x.shape[0], 32, 7, 7).to(device)
        s6 = torch.zeros(x.shape[0], 64, 7, 7).to(device)
        v6 = torch.zeros(x.shape[0], 64, 7, 7).to(device)

        s7 = torch.zeros(x.shape[0], self.Nout).to(device)
        v7 = torch.zeros(x.shape[0], self.Nout).to(device)
        nsp = torch.zeros(x.shape[0], self.Nout).to(device)

        for i in range(self.Nsp):
            xi = poisson_spikes(x,self.scale).to(device)

            xi = self.conv1(xi)
            s1, v1 = self.sl1(xi, s1, v1)
            s1 = nn.ReLU()(s1)

            xi = self.conv2(s1)
            s2, v2 = self.sl2(xi, s2, v2)
            s2 = nn.ReLU()(s2)

            s2 = nn.MaxPool2d(kernel = (2, 2))(s2)

            xi = self.conv3(s2)
            s3, v3 = self.sl3(xi, s3, v3)
            s3 = nn.ReLU()(s3)

            xi = self.conv4(s3)
            s4, v4 = self.sl4(xi, s4, v4)
            s4 = nn.ReLU()(s4)

            s4 = nn.MaxPool2d(kernel = (2, 2))(s4)

            xi = self.conv1(xi)
            s5, v5 = self.sl5(xi, s5, v5)
            s5 = nn.ReLU()(s5)

            xi = self.conv2(s5)
            s6, v6 = self.sl6(xi, s6, v6)
            s6 = nn.ReLU()(s6)

            s6 = nn.MaxPool2d(kernel = (2, 2))(s6)

            xi = s6.view(s6.shape[0],-1)
            xi2 = self.l1(xi)
            s7, v7 = self.sl7(xi2, s7, v7)
            nsp += s3
        return nsp