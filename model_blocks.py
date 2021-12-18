# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F


class SeparableConv2d(nn.Module):
    """
        SeparableConv2d is provided out-of-the-box in keras;
        this is the PyTorch implementation suggested here:
        https://stackoverflow.com/a/65155106/1616037
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        # a regular convolution (with the arbitrary kernel size),
        # which is applied to each channel separately
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, groups=in_channels, bias=bias, padding=1)

        # a convolution with a (1, 1)-kernel which converts
        # the feature map to another one with the required number of `out_channels`
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=(1, 1), bias=bias)
        # ...so essentially this is a 'factorization' of 'heavier' convolutions

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class FirstBlock(nn.Module):
    """ Two regular convolution blocks. Input image -- to a feature map. """

    def __init__(self, in_channels=1,
                 conv_out=64, conv_kernel_size=(3, 3), conv_stride=(1, 1),
                 mp_kernel_size=(3, 3), mp_stride=(2, 2)):
        super(FirstBlock, self).__init__()

        # a regular convolution block: conv2d + batch_norm + max_pooling
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv_out,
                               kernel_size=conv_kernel_size, stride=conv_stride, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_out)
        self.mp1 = nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride, padding=1)

        # ...then ReLU

        # ...then yet another regular convolution block: conv2d + batch_norm + max_pooling
        self.conv2 = nn.Conv2d(in_channels=conv_out, out_channels=conv_out,
                               kernel_size=conv_kernel_size, stride=conv_stride, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_out)
        self.mp2 = nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride, padding=1)

        # ...then ReLU.

    def forward(self, images_tensor):
        first_pass_output = F.relu(self.mp1(self.bn1(self.conv1(images_tensor))))
        second_pass_output = F.relu(self.mp2(self.bn2(self.conv2(first_pass_output))))
        return second_pass_output


class InnerBlock(nn.Module):
    """ Inception-like block: separable convolutions, batch-normalizations, activations and max-pooling """

    def __init__(self, in_channels, sconv_out=128, sconv_kernel_size=(3, 3), mp_kernel_size=(3, 3), mp_stride=(2, 2)):
        super(InnerBlock, self).__init__()

        self.sconv1 = SeparableConv2d(in_channels=in_channels, out_channels=sconv_out, kernel_size=sconv_kernel_size)
        self.bn1 = nn.BatchNorm2d(sconv_out)
        self.sconv2 = SeparableConv2d(in_channels=sconv_out, out_channels=sconv_out, kernel_size=sconv_kernel_size)
        self.bn2 = nn.BatchNorm2d(sconv_out)
        self.mp = nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride, padding=1)

    def forward(self, x):
        first_pass_output = F.relu(self.bn1(self.sconv1(x)))
        second_pass_output = F.relu(self.mp(self.bn2(self.sconv2(first_pass_output))))
        return second_pass_output


class FinalBlock(nn.Module):
    """
        The block of GlyphNet preparing the outputs:
        separable convolution + global average pooling + dropout + MLP + softmax"""

    def __init__(self, in_channels=256, out_size=172, sconv_out=512, sconv_kernel_size=(3, 3), dropout_rate=0.15):

        super(FinalBlock, self).__init__()
        self.sconv = SeparableConv2d(in_channels=in_channels, out_channels=sconv_out, kernel_size=sconv_kernel_size)
        self.bn = nn.BatchNorm2d(sconv_out)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fully_connected = nn.Linear(in_features=sconv_out, out_features=out_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor):
        sconv_pass_result = F.relu(self.bn(self.sconv(input_tensor)))

        # computing average over each feature map
        pooled = torch.mean(sconv_pass_result, dim=(-1, -2)) # global average pooling
        return self.softmax(self.fully_connected(self.dropout(pooled)))
