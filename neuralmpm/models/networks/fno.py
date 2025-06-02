from typing import Callable, List

import torch
from torch import nn


class SpectralConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.kernels = nn.Parameter(
            self.scale
            * torch.rand(
                2,
                self.modes,
                self.modes,
                in_channels,
                out_channels,
                dtype=torch.cfloat,
            )
        )

    def forward(self, x):
        x_hat = torch.fft.rfft2(x, dim=(-3, -2))
        out_ft = torch.zeros(
            *x_hat.shape[:-1], self.out_channels, dtype=torch.cfloat, device=x.device
        )
        out_ft[:, : self.modes, : self.modes, :] = torch.einsum(
            "bxyi,xyio->bxyo", x_hat[:, : self.modes, : self.modes, :], self.kernels[0]
        )
        out_ft[:, -self.modes :, : self.modes, :] = torch.einsum(
            "bxyi,xyio->bxyo", x_hat[:, -self.modes :, : self.modes, :], self.kernels[1]
        )
        out = torch.fft.irfft2(out_ft, s=(x.size(-3), x.size(-2)), dim=(-3, -2))

        return out


class FNOBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        act: Callable = nn.ReLU(),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.act = act
        self.spectral_conv = SpectralConv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            modes=self.modes,
        )
        self.bypass_conv = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):
        spectral_out = self.spectral_conv(x)
        bypass_out = self.bypass_conv(x)
        out = self.act(spectral_out + bypass_out)
        return out


# TODO: Include share_params like FFNO here.
class FNO(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        num_preds: int,
        out_channels: int,
        modes: int,
        use_mlp: bool = False,
        act: nn.Module = nn.LeakyReLU(),
    ):
        super().__init__()
        self.modes = modes
        self.num_preds = num_preds
        self.in_channels = in_channels
        self.out_channels = out_channels

        if use_mlp:
            self.lifting = nn.Sequential(
                nn.Linear(in_channels, 2 * hidden_channels[0]),
                act,
                nn.Linear(2 * hidden_channels[0], hidden_channels[0]),
                act,
            )
        else:
            self.lifting = nn.Linear(in_channels, hidden_channels[0])

        self.blocks = nn.ModuleList()
        for in_channels_f, out_channels_f in zip(
            hidden_channels[:-1], hidden_channels[1:]
        ):
            self.blocks.append(
                FNOBlock(
                    in_channels=in_channels_f,
                    out_channels=out_channels_f,
                    modes=self.modes,
                    act=act,
                )
            )

        self.projections = nn.ModuleList()
        for i in range(out_channels):
            if use_mlp:
                self.projections.append(
                    nn.Sequential(
                        nn.Linear(hidden_channels[-1], 2 * hidden_channels[-1]),
                        act,
                        nn.Linear(2 * hidden_channels[-1], num_preds),
                    )
                )
            else:
                self.projections.append(nn.Linear(hidden_channels[-1], num_preds))

    def forward_blocks(self, x):
        for fno in self.blocks:
            x = fno(x)
        return x

    def forward(self, x):
        x = self.lifting(x)
        x = self.forward_blocks(x)

        x = torch.stack([proj(x) for proj in self.projections], dim=-1)

        # x = x.view(*x.shape[:-1], self.num_preds, self.out_channels)

        return x
