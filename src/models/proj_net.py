import numpy as np
# 3p
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary

class ProjNet(nn.Module):
    def __init__(
        self, 
        **kwargs):
        super(ProjNet, self).__init__()
        self.version = kwargs['version']

        self.optimizer = kwargs['optimizer']
        self.learning_rate = kwargs['learning_rate']
        self.batch_size = kwargs['batch_size']

        self.in_shape = kwargs['in_shape']
        self.conv_channels = kwargs['conv_channels']

        self.n_last_conv_layer = self.conv_channels[-1]*np.product(self.in_shape[1:])

        self.conv = nn.Sequential()

        self.in_ = self.in_shape[0]
        for layer, width in enumerate(self.conv_channels):
            self.conv.add_module(f'conv_{layer}',nn.Conv2d(
                in_channels=self.in_, 
                out_channels=width, 
                kernel_size=3, 
                padding=1))
            self.conv.add_module(f'relu_{layer}', nn.ReLU())
            self.conv.add_module(f'bn_{layer}', nn.BatchNorm2d(width))
            self.in_ = width
            layer_count = layer

        self.fc_channels = kwargs['fc_channels']
        self.dropout = kwargs['dropout']
        self.fc = nn.Sequential()

        self.in_ = self.n_last_conv_layer
        for width, dropout_ in zip(self.fc_channels, self.dropout):
            self.fc.add_module(f'fc_{layer_count}', nn.Linear(
                in_features=self.in_, 
                out_features=width))
            self.in_ = width
            layer_count += 1
            self.fc.add_module(f'relu_{layer}', nn.ReLU())
        if 0.0 < dropout_ < 1.0:
            self.fc.add_module(f'drop_{layer_count}', nn.Dropout(dropout_))
        else:
            raise ValueError('Given dropout rate must be between 0 and 1')

        self.fc.add_module(
            f'fc_{layer_count+1}',
            nn.Linear(in_features=self.fc_channels[-1], out_features=1))

    def forward(self, x):
        # implement the forward pass
        x = self.conv.forward(x)
        x = x.view(-1, self.n_last_conv_layer) 
        x = self.fc.forward(x)  
        if torch.isnan(x).any():
            warnings.warn('NaN value in prediction')
        x = torch.nan_to_num(x, nan=0.0)              
        return x

    # def summary(self):
    #     summary(self, (1, *self.in_shape))

    def get_version(self):
        return self.version

    def get_optimizer(self):
        return self.optimizer

    def get_learning_rate(self):
        return self.learning_rate

    def get_batch_size(self):
        return self.batch_size
