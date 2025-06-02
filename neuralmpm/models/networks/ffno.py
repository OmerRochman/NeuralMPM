from typing import List

import torch
from torch import nn

from neuralmpm.models.networks.fno import FNO

# TODO fix and make it alike FNO


class FactorizedSpectralConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.kernels = nn.Parameter(
            self.scale
            * torch.rand(
                2,  # x, y.
                self.modes,
                in_channels,
                out_channels,
                dtype=torch.cfloat,
            )
        )

    def forward(self, x):
        # Fourier over X dim.
        x_hat_x = torch.fft.rfft(x, dim=-3, norm="ortho")  # [B, X//2+1, Y, C]
        out_ft_x = torch.zeros(
            *x_hat_x.shape[:-1], self.out_channels, dtype=torch.cfloat, device=x.device
        )

        out_ft_x[:, : self.modes, :, :] = torch.einsum(
            "bxyi,xio->bxyo", x_hat_x[:, : self.modes, :, :], self.kernels[0]
        )

        out_x = torch.fft.irfft(out_ft_x, dim=-3, norm="ortho")

        # Fourier over Y dim.
        x_hat_y = torch.fft.rfft(x, dim=-2, norm="ortho")  # [B, X, Y//2+1, C]
        out_ft_y = torch.zeros(
            *x_hat_y.shape[:-1], self.out_channels, dtype=torch.cfloat, device=x.device
        )
        out_ft_y[:, :, : self.modes, :] = torch.einsum(
            "bxyi,yio->bxyo", x_hat_y[:, :, : self.modes, :], self.kernels[1]
        )

        out_y = torch.fft.irfft(out_ft_y, dim=-2, norm="ortho")

        return out_x + out_y


class FFNOBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        act: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.act = act
        self.spectral_conv = FactorizedSpectralConv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            modes=self.modes,
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.out_channels, 4 * self.out_channels),
            self.act,
            nn.Linear(4 * self.out_channels, self.out_channels),
        )

    def forward(self, x):
        out = self.spectral_conv(x)
        out = self.ffn(out)
        return x + out


class FFNO(FNO):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        num_preds: int,
        out_channels: int,
        modes: int,
        share_params: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        super().__init__(
            in_channels, hidden_channels, num_preds, out_channels, modes, act
        )

        self.share_params = share_params
        self.blocks = nn.ModuleList()
        self.num_blocks = len(hidden_channels) - 1
        if not share_params:
            for in_channels_f, out_channels_f in zip(
                hidden_channels[:-1], hidden_channels[1:]
            ):
                self.blocks.append(
                    FFNOBlock(
                        in_channels=in_channels_f,
                        out_channels=out_channels_f,
                        modes=self.modes,
                        act=act,
                    )
                )
        else:
            self.blocks.append(
                FFNOBlock(
                    in_channels=hidden_channels[0],
                    out_channels=hidden_channels[0],
                    modes=self.modes,
                    act=act,
                )
            )

    def forward_blocks(self, x):
        if not self.share_params:
            return super().forward_blocks(x)

        fno = self.blocks[0]
        for _ in range(self.num_blocks):
            x = fno(x)

        return x
