import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange

"""
Imitation learning network
"""
def double_conv(in_channels, out_channels, mid_channels=None):
    mid_channels = out_channels if mid_channels is None else mid_channels

    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, 3, padding=1),
        # nn.GroupNorm(1, mid_channels),
        nn.BatchNorm2d(mid_channels),
        nn.GELU(),
        nn.Conv2d(mid_channels, out_channels, 3, padding=1),
        # nn.GroupNorm(mid_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.GELU()
    )

class CNN(nn.Module):

    def __init__(self, history_length=1, n_classes=3): 
        super(CNN, self).__init__()
        # (96, 96, history_length) -> double conv -> (96, 96, 32) -> maxpool -> double conv -> (48, 48, 64) -> maxpool -> 
        # double conv -> (24, 24, 128) -> maxpool -> double conv -> (12, 12, 256) -> maxpool -> double conv -> (6, 6, 512) -> maxpool
        # Linear : -> 3 * 3 * 512 -> 100 -> 3
        self.conv1 = double_conv(history_length, 32, 16)
        self.conv2 = double_conv(32, 64)
        self.conv3 = double_conv(64, 128)
        self.conv4 = double_conv(128, 256)
        self.conv5 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.linear1 = nn.Linear(3 * 3 * 512, 100)
        self.linear2 = nn.Linear(100, n_classes)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        # x = torch.transpose(x, (0, 3, 1, 2))
        x = rearrange(x, "b h w c -> b c h w")
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.maxpool(self.conv3(x))
        x = self.maxpool(self.conv4(x))
        x = self.maxpool(self.conv5(x))
        # reshape always copies memory. view never copies memory.
        x = x.view(x.shape[0], -1)
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return x

def double_conv2(in_channels, out_channels, mid_channels=None):
    mid_channels = out_channels if mid_channels is None else mid_channels

    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, 3),
        # nn.GroupNorm(1, mid_channels),
        nn.BatchNorm2d(mid_channels),
        nn.GELU(),
        nn.Conv2d(mid_channels, out_channels, 3, stride=2),
        # nn.GroupNorm(mid_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.GELU()
    )

class CNN2(nn.Module):

    def __init__(self, history_length=1, n_classes=3): 
        super(CNN2, self).__init__()
        self.conv1 = double_conv2(history_length, 32, 16)
        self.conv2 = double_conv2(32, 64)
        self.conv3 = double_conv2(64, 128)
        self.conv4 = double_conv2(128, 256, 256)
        self.linear1 = nn.Linear(256 * 3 * 3, 256)
        self.linear2 = nn.Linear(256, n_classes)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        # x = torch.transpose(x, (0, 3, 1, 2))
        x = rearrange(x, "b h w c -> b c h w")
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        # reshape always copies memory. view never copies memory.
        x = x.view(x.shape[0], -1)
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return x

class CNN3(nn.Module):

    def __init__(self, history_length=1, n_classes=3): 
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv2d(history_length, 16, 3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2)
        self.linear1 = nn.Linear(5 * 5 * 128, n_classes)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        # x = torch.transpose(x, (0, 3, 1, 2))
        x = rearrange(x, "b h w c -> b c h w")
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        # reshape always copies memory. view never copies memory.
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        return x
    

class CNN4(nn.Module):

    def __init__(self, history_length=1, n_classes=3): 
        super(CNN4, self).__init__()
        self.conv1 = nn.Conv2d(history_length, 8, 3, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2)
        self.linear1 = nn.Linear(11 * 11 * 32, n_classes)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        # x = torch.transpose(x, (0, 3, 1, 2))
        if len(x.shape) != 4:
            x = x[..., None]
        x = rearrange(x, "b h w c -> b c h w")
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = self.act(self.conv4(x))
        # reshape always copies memory. view never copies memory.
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        return x

# class CNN5


