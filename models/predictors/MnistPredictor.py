# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:20:12 2021

BIC Group Project - MNIST Predictor

@author: Woo Jia Jun

30/7 0135 Update:
- Added global variable flag to denote use of spiking CNN model
- Normalisation implemented (potentially naive solution)
- Quantization still to be done.

"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.utils.prune as prune  # for weight pruning
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# import modules from this project
from MnistLoader import train_loader, test_loader
from SimpleCNN import SimpleCNN
from SpikingCNN import SpikingCNN

activate_spiking = False  # if true, use spiking CNN model
net = (SpikingCNN() if activate_spiking else SimpleCNN())  # initialise the CNN model
n_epochs = 10  # number of training epochs
batch_size_train = 64  # training set batch size
batch_size_test = 1000  # testing set batch size
learning_rate = 0.01
momentum = 0.5
log_interval = 10

# random seeding
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# pruning-related variables
activate_prune = False # if true, apply pruning during the train-test process
online_prune = False  # if true, apply prune process during training process
prune_method = "random"  # specify the method to determine weights to prune

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch: int):
    """
    Trains the neural network for the given number of epochs.

    Parameters
    ----------
    epoch : int
        The number of intended training iterations.

    Returns
    -------
    None.

    """
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # conduct loss normalisation
        height = loss.size()[-2]
        width = loss.size()[-1]
        unnorm_loss = loss.view(loss.size(0), -1)
        unnorm_loss -= unnorm_loss.min(1, keepdim=True)[0]
        unnorm_loss /= unnorm_loss.max(1, keepdim=True)[0]
        norm_loss = unnorm_loss.view(batch_size_train, height, width)
        norm_loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(net.state_dict(), '/results/model.pth')
            torch.save(optimizer.state_dict(), '/results/optimizer.pth')


def test():
    """
    Function to test a trained neural network.

    Returns
    -------
    None.

    """
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = net(data)
            loss = F.nll_loss(output, target, size_average=False)
            # conduct loss normalisation (Note: loss here appears to be a
            # scalar, unsure if methodology is correct)
            height = loss.size()[-2]
            width = loss.size()[-1]
            unnorm_loss = loss.view(loss.size(0), -1)
            unnorm_loss -= unnorm_loss.min(1, keepdim=True)[0]
            unnorm_loss /= unnorm_loss.max(1, keepdim=True)[0]
            norm_loss = unnorm_loss.view(batch_size_test, height, width)
            test_loss += norm_loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def weight_prune(method: str = "random", prune_percentage: float = 0.85):
    """
    Prune weights from the model to compress the neural network and reduce
    computational costs of network operation.

    Parameters
    ----------
    method : str, optional
        Denotes the method by which weights are to be pruned. Available
        methods are random selection ("random") and pruning weights based on
        lowest L1-norm ("L1").
        The default is "random".
    prune_percentage : float, optional
        The fraction of weights to be pruned, expressed as a decimal float.
        The default is 0.85.

    Returns
    -------
    None.

    """
    if method == "random":
        prune.random_unstructured(net, "weight", prune_percentage)
    elif method == "L1":
        prune.l1_unstructured(net, "weight", prune_percentage)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

test()
if (activate_prune and online_prune):
    # in the paper, the online prune method requires a short training set
    # of 30,000 images. The below train is a crude mimicry of this initial
    # training.
    train(1)
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
    if (activate_prune and online_prune):
        # for online weight pruning,
        weight_prune(prune_method)
if (activate_prune and not(online_prune)):
    # for post-training weight pruning, one pruning process is executed
    # after the whole training process.
    weight_prune(prune_method)
