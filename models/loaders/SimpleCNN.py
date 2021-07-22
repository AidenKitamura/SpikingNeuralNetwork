import torch.nn as nn


class SimpleCNN(nn.Module):
    """ Class Summary
    This SimpleCNN backbone contains 6 layers,
    each of which uses a ReLU activation and a
    3*3 kernel. A max pooling of size 2*2 is 
    applied after every 2 CNN layers
    
    Parameters:
    - in_channel: input data channels
    - output_dim: latent size of linear layer
      to be fed to the predictor
    - input_size: the input dimension of the 
      image, in the form of W*H
    """

    def __init__(self, in_channels, input_size, output_dim=256, **args):
        super(SimpleCNN, self).__init__()
        self.args = args
        self.output_dim = output_dim
        self.input_size = input_size
        for iteration_num in range(3):
            self.input_size = (int(self.input_size[0] / 2),\
                int(self.input_size[1] / 2))
        self.output_size = self.input_size
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 2,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels = 2, out_channels = 4,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2)),
            nn.Conv2d(in_channels = 4, out_channels = 8,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels = 8, out_channels = 16,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2)),
            nn.Conv2d(in_channels = 16, out_channels = 32,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64,\
                kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2)),
            nn.Flatten(),
            nn.Linear(64 * self.output_size[0] * self.output_size[1],\
                output_dim)
        )

    def forward(self, x):
        out = self.model(x)
        return out