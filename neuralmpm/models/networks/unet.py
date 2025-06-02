from typing import List

import torch
from torch import nn


class MLP(nn.Sequential):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        architecture: List,
        activation=torch.nn.LeakyReLU,
    ):
        super(MLP, self).__init__()

        architecture = list(architecture)

        layers = []

        for current_size, next_size in zip(
            [num_inputs] + architecture, architecture + [num_outputs]
        ):
            layers.extend(
                [
                    torch.nn.Linear(current_size, next_size),
                    activation(),
                ]
            )

        layers = layers[:-1]
        layers = filter(lambda layer: layer is not None, layers)

        super().__init__(*layers)

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, strides=1, act=nn.LeakyReLU
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=kernel_size // 2,
        )
        self.act = act()

    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        return out


class DownsampleBlock(nn.Module):
    def __init__(self, out_channels, factor=2, act=nn.LeakyReLU):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=factor * 2,
            stride=[factor] * 2,
            padding=factor // 2,
        )
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, out_channels, factor=2, act=nn.LeakyReLU):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=[factor * 2] * 2,
            stride=[factor] * 2,
            padding=[factor // 2] * 2,
        )
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        architecture,
        factors,
        in_channels,
        kernel_size=3,
        strides=1,
        out_channels=3,
    ):
        super(UNet, self).__init__()

        self.encoder = MLP(
            num_inputs=in_channels,
            num_outputs=architecture[0],
            architecture=[64, 64, 64],
        )
        self.downsampling_blocks = nn.ModuleList(
            [
                DownsampleBlock(f, factor=fact)
                for f, fact in zip(architecture[:-1], factors)
            ]
        )
        self.upsampling_blocks = nn.ModuleList(
            [
                UpsampleBlock(f, factor=fact)
                for f, fact in zip(reversed(architecture[:-1]), reversed(factors))
            ]
        )

        self.conv_down_blocks = nn.ModuleList(
            [
                ConvBlock(f, f, kernel_size=kernel_size, strides=strides)
                for f in list(architecture[:-1]) + [architecture[-2]]
            ]
        )
        self.conv_up_blocks = nn.ModuleList(
            [
                ConvBlock(2 * f, f, kernel_size=kernel_size, strides=strides)
                for f in [architecture[-2]] + list(reversed(architecture[:-1]))
            ]
        )

        self.decoders = nn.ModuleList(
            [
                MLP(
                    architecture=[64, 64, 64],
                    num_outputs=architecture[-1],
                    num_inputs=architecture[-2],
                )
                for _ in range(out_channels)
            ]
        )

    def forward(self, x, particles=None):
        skip_connections = []
        # Encoder

        x = self.encoder(x).permute(0, -1, 1, 2)

        for conv, down in zip(self.conv_down_blocks[:-1], self.downsampling_blocks):
            x = conv(x)
            skip_connections.append(x)
            x = down(x)

        # Bottleneck

        x = self.conv_down_blocks[-1](x)

        # Decoder
        for up, conv, skip in zip(
            self.upsampling_blocks, self.conv_up_blocks, reversed(skip_connections)
        ):
            x = up(x)

            x = torch.cat([x, skip], axis=1)  # Skip connection

            x = conv(x)

        x = x.permute(0, 2, 3, 1)

        x = torch.stack(
            [decoder(x) for decoder in self.decoders],
            dim=-1,
        )

        return x
