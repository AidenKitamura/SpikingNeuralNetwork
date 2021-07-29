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

def train(args, model, device, train_loader, optimizer, epoch, scale=4):
    model.train()
    Nsp = model.Nsp
    for batch_idx, (data, target) in enumerate(train_loader):
        bsize = target.shape[0]
        optimizer.zero_grad()
        mtarget = target
        mdata = data
        data, mtarget = mdata.to(device), mtarget.to(device)
        output = scale*(model(data, device)-0.5*model.Nsp)
        loss = F.cross_entropy(output, mtarget)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#            print(output)

def test(args, model, device, test_loader, scale=4):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            mtarget = target
            mdata = data
            data, mtarget = mdata.to(device), mtarget.to(device)
            output = scale*(model(data, device)-0.5*model.Nsp)
            test_loss += F.cross_entropy(output, mtarget).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).to(device)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def train_mse(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        bsize = target.shape[0]
        optimizer.zero_grad()
        mtarget = torch.zeros(target.shape[0],10)
        for i in range(target.shape[0]):
            mtarget[i,target[i]]= args.spikes
        data, target = data.to(device), mtarget.to(device)
        output = model(data, device)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#            print(output)

def test_mse(args, model, device, test_loader):
    model.eval()
    Nst = model.Nsp
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            mtarget = torch.zeros(target.shape[0],10)
            for i in range(target.shape[0]):
                mtarget[i,target[i]]= args.spikes
            data, mtarget = data.to(device), mtarget.to(device)
            output = model(data, device)
            test_loss += F.mse_loss(output, mtarget, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred).to(device)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='SpikingNet example')
    parser.add_argument('name', metavar='N', type=str, nargs=1,
                        help='filename')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dataset', type=int, default=0, metavar='N',
                        help='dataset: mnist-0 fashionmnist-1')
    parser.add_argument('--length', type=int, default=8, metavar='N',
                        help='length: (default: 8)')
    parser.add_argument('--leakage1', type=int, default=4, metavar='N',
                        help='leakage2: (default: 4)')
    parser.add_argument('--leakage2', type=int, default=4, metavar='N',
                        help='leakage1: (default: 4)')
    parser.add_argument('--leakage0', type=int, default=4, metavar='N',
                        help='leakage0: (default: 4)')
    parser.add_argument('--beta', type=float, default=5.0, metavar='N',
                        help='beta: (default: 5.0)')
    parser.add_argument('--scale', type=float, default=1.0, metavar='N',
                        help='scale: (default: 1.0)')
    parser.add_argument('--cost', type=int, default=1, metavar='N',
                        help='cost function 0 - xent, 1 - mse: (default: 1)')
    parser.add_argument('--spikes', type=int, default=4, metavar='N',
                        help='# output spikes in mse: (default: 4)')
    parser.add_argument('--model', type=int, default=0, metavar='N',
                        help='model: shallow-0 hidden-1 conv1-2 conv2-3 \
                            Lenet5-4 Lenet5const-5')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.dataset == 1:
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('fashionMNIST', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('fashionMNIST', train=False, transform=transforms.Compose([
                               transforms.ToTensor()])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('MNIST', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('MNIST', train=False, transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)


    if args.model == 0:
        model = SpikingShallowNetwork(784, 10, args.length, args.leakage1,
            beta=args.beta, scale=args.scale).to(device)
    elif args.model == 1:
        model = SpikingHiddenNetwork(784, 10, 30, args.length, args.leakage1,
            args.leakage2, beta=args.beta, scale=args.scale).to(device)
    elif args.model == 2:
        model = SpikingConvNetwork(784, 10, args.length, args.leakage1,
            args.leakage2, beta=args.beta, scale=args.scale).to(device)
    elif args.model == 3:
        model = SpikingConvNetwork2(784, 10, args.length, args.leakage1,
            args.leakage2, beta=args.beta, scale=args.scale).to(device)
    elif args.model == 4:
        model = SpikingLeNet5(args.length, args.leakage1, args.leakage2,
            beta=args.beta, scale=args.scale).to(device)
    elif args.model == 5:
        model = SpikingLeNet5const(args.length, args.leakage0, args.leakage1,
            args.leakage2, beta=args.beta, scale=args.scale).to(device)



    if args.cost == 0:
        trainf = train
        testf = test
    else:
        trainf = train_mse
        testf = test_mse

    optimizer = optim.Adam(model.parameters(), lr=args.lr*8/args.length)
    data = []

    for epoch in range(1, args.epochs + 1):
        trainf(args, model, device, train_loader, optimizer, epoch)
        result = testf(args, model, device, test_loader)
        data.append([epoch, result])

    data = np.array(data)
#    condstring = "%d_%d_%d_%d_%d" % (args.length, args.leakage0, args.leakage1, args.leakage2, args.dataset)
    condstring = "%d_%d_%d_%d" % (args.length, args.leakage1, args.beta, args.dataset)

    filename = args.name[0] + "_" + condstring + ".npy"
    filemode = args.name[0] + "_" + condstring + ".pt"
    np.save(filename, data)
    torch.save(model.state_dict(), filemode)

if __name__ == '__main__':
    main()