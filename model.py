# coding: utf-8

from torch import nn


class SeparableConv2d(nn.Module):
    """
        SeparableConv2d is provided out-of-the-box in keras;
        this is the implementation suggested here: https://stackoverflow.com/a/65155106/1616037
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

        # ...so essentially this is a 'factorization' of heavier convolutions

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Glyphnet(nn.Module):

    def __init__(self):
        super().__init__()
        # todo:

    def forward(self, image_input_tensor):
        raise NotImplementedError()


if __name__ == "__main__":
    import torchvision.models as models