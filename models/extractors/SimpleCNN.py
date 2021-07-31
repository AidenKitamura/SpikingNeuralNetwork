import torch.nn as nn


class SimpleCNN(nn.Module):
    """ Class Summary
    This SimpleCNN backbone contains 4 layers,
    each of which uses a ReLU activation and a
    3*3 kernel. A max pooling of size 2*2 is 
    applied after every CNN layer. A dropout
    layer is applied at the last layer
    
    Parameters:
    - in_channel: input data channels
    - output_dim: latent size of linear layer
      to be fed to the predictor
    - input_size: the input dimension of the 
      image, in the form of W*H
    """

    def __init__(self, in_channels, output_dim=256, **args):
        super(SimpleCNN, self).__init__()
        self.args = args
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 4,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 16,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 64,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 256,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.dropout = nn.Dropout(p = 0.9)
        self.linear = nn.Linear(256, output_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = nn.ReLU()(out)
        out = nn.MaxPool2d(kernel_size = (2, 2))
        out = self.conv2(out)
        out = nn.ReLU()(out)
        out = nn.MaxPool2d(kernel_size = (2, 2))
        out = self.conv3(out)
        out = nn.ReLU()(out)
        out = nn.MaxPool2d(kernel_size = (2, 2))
        out = self.conv4(out)
        out = nn.ReLU()(out)
        out = nn.MaxPool2d(kernel_size = (2, 2))
        out = nn.Flatten()(out)
        out = self.dropout(out)
        out = self.linear(out)
        out = nn.Softmax(dim = 0)(out)

        return out