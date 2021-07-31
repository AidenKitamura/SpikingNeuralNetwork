import torch
import torchvision
from extractors.modules import SpikingActivation, TemporalAvgPool
import torch.nn as nn


class SpikingCNN(nn.Module):
    """ Class Summary
    This SpikingCNN backbone contains 6 layers,
    each of which uses a ReLU activation and a
    3*3 kernel. A max pooling of size 2*2 is 
    applied after every 2 CNN layers. A drop-
    out layer is applied at the last layer
    
    Parameters:
    - in_channel: input data channels
    - output_dim: latent size of linear layer
      to be fed to the predictor
    - input_size: the input dimension of the 
      image, in the form of W*H
    """

    def __init__(self, in_channels, output_dim = 10, frequency = 10, non_spiking_model = None, **args):
        super(SpikingCNN, self).__init__()
        self.args = args
        self.output_dim = output_dim
        self.frequency = frequency
        self.dt = 1 / self.frequency

        if non_spiking_model == None:
            self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 4,\
                    kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
            self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 16,\
                    kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
            self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 64,\
                    kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
            self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 256,\
                    kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        else:
            self.conv1 = non_spiking_model.conv1
            self.conv2 = non_spiking_model.conv2
            self.conv3 = non_spiking_model.conv3
            self.conv4 = non_spiking_model.conv4

        self.dropout = nn.Dropout(p = 0.9)
        self.linear = nn.Linear(256, output_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = SpikingActivation(nn.ReLU(), dt = self.dt, spiking_aware_training = False)
        out = nn.MaxPool2d(kernel_size = (2, 2))
        out = self.conv2(out)
        out = SpikingActivation(nn.ReLU(), dt = self.dt, spiking_aware_training = False)
        out = nn.MaxPool2d(kernel_size = (2, 2))
        out = self.conv3(out)
        out = SpikingActivation(nn.ReLU(), dt = self.dt, spiking_aware_training = False)
        out = nn.MaxPool2d(kernel_size = (2, 2))
        out = self.conv4(out)
        out = SpikingActivation(nn.ReLU(), dt = self.dt, spiking_aware_training = False)
        out = nn.MaxPool2d(kernel_size = (2, 2))
        out = nn.Flatten()(out)
        out = self.dropout(out)
        out = self.linear(out)
        out = nn.Softmax(dim = 0)(out)

        return out