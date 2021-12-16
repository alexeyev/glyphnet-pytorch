# coding: utf-8

from torch import nn
from torch.nn import Sequential
import logging

from model_blocks import FirstBlock, InnerBlock, FinalBlock


class Glyphnet(nn.Module):
    """
        PyTorch GlyphNet hieroglyph classifier implementation.

        Some hyperparameters are hardcoded, otherwise the whole configuration
        would have been passed as a dict or a long sequence of parameters to the __init__ method,
        which may be ugly yet the right thing to do; I decided I won't do that.
    """

    def __init__(self, in_channels=1,
                 num_classes=172,
                 first_conv_out=64,
                 last_sconv_out=512,
                 sconv_seq_outs=(128, 128, 256, 256)):
        """
        :param in_channels: the number of channels in the input image
        :param num_classes: the number of labels for prediction
        :param first_conv_out:
        :param last_sconv_out:
        :param sconv_seq_outs:
        """

        super(Glyphnet, self).__init__()
        self.first_block = FirstBlock(in_channels, first_conv_out)
        in_channels_sizes = [first_conv_out] + list(sconv_seq_outs)
        self.inner_blocks = Sequential(*(InnerBlock(in_channels=i, sconv_out=o)
                                         for i, o in zip(in_channels_sizes, sconv_seq_outs)))
        self.final_block = FinalBlock(in_channels=in_channels_sizes[-1], out_size=num_classes, sconv_out=last_sconv_out)

    def forward(self, image_input_tensor):
        x = self.first_block(image_input_tensor)
        x = self.inner_blocks(x)
        x = self.final_block(x)

        return x


if __name__ == "__main__":
    import torch

    model = Glyphnet()

    print("...the proposed network has a much lower number of parameters, "
          "which is only 498856 (of which 494504 are trainable), compared to the...")

    print("Total:", sum(p.numel() for p in model.parameters()))
    print("Trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    dummy_input = torch.zeros((256, 1, 100, 100))  # batch, single-channel-image
    result = model(dummy_input)

    print(result.shape)
