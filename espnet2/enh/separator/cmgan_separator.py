from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from espnet2.enh.layers.mp_senet import (
    DeConvBlock,
    DenseBlock,
    DenseEncoder,
    TSConformerBlock,
)
from espnet2.enh.separator.abs_separator import AbsSeparator


class MaskDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hid_chans: int = 64,
        ksz: tuple = (1, 3),
        stride: tuple = (1, 2),
        depth: int = 4,
    ):
        super(MaskDecoder, self).__init__()
        self.dense_net = DenseBlock(
            in_channels, hid_chans, hid_chans, ksz, depth
        )
        self.de_conv = DeConvBlock(hid_chans, hid_chans, ksz, stride)
        self.conv = nn.Sequential(
            nn.Conv2d(hid_chans, out_channels, (1, 1)),
            nn.InstanceNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (1, 1)),
        )
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        # x shape: (B, C, T, F)
        x = self.dense_net(x)
        # x shape: (B, C, T, 2 x F)
        x = self.de_conv(x)
        # x shape: (B, C', T, 2 x F)
        x = self.conv(x)
        # x shape: (B, C', T, 2 x F)
        x = self.prelu(x)
        return x


class ComplexDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        hid_chans: int = 64,
        ksz: tuple = (1, 3),
        stride: tuple = (1, 2),
        depth: int = 4,
    ):
        super(ComplexDecoder, self).__init__()
        self.dense_net = DenseBlock(
            in_channels, hid_chans, hid_chans, ksz, depth
        )
        self.de_conv = DeConvBlock(hid_chans, hid_chans, ksz, stride)
        self.conv = nn.Sequential(
            nn.Conv2d(hid_chans, out_channels, (1, 1)),
            nn.InstanceNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (1, 1)),
        )

    def forward(self, x):
        # x shape: (B, C, T, F)
        x = self.dense_net(x)
        # x shape: (B, C, T, 2 x F)
        x = self.de_conv(x)
        # x shape: (B, C', T, 2 x F)
        x = self.conv(x)
        return x


class CMGANSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim,
        num_spk: int = 1,
        in_channels: int = 3,
        hid_chans: int = 64,
        num_tsblocks: int = 4,
    ):
        super(CMGANSeparator, self).__init__()
        self._num_spk = num_spk
        self.num_tsblocks = num_tsblocks
        self.enc = DenseEncoder(in_channels, hid_chans, hid_chans)

        self.ts_conformer = nn.ModuleList([])
        for i in range(num_tsblocks):
            self.ts_conformer.append(TSConformerBlock(hid_chans))

        self.mask_dec = MaskDecoder(hid_chans, num_spk, hid_chans)
        self.complex_dec = ComplexDecoder(hid_chans, 2 * num_spk, hid_chans)

    def forward(
        self,
        input: torch.Tensor,
        ilen: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """forward function for CMGAN

        Args:
            input (torch.Tensor): STFT encoded feature, [Batch, T, F].
            ilen (torch.Tensor): input lengths, [Batch,].
            additional (Optional[Dict]): other data. Currently not used in this model.

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(Batch, T, F), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
        """
        noisy_mag, noisy_phase, noisy_real, noisy_imag = (
            torch.abs(input),
            torch.angle(input),
            input.real,
            input.imag,
        )
        # (B, T, F) -> (B, 3, T, F)
        x = torch.stack((noisy_mag, noisy_real, noisy_imag), dim=1)
        # (B, 3, T, F) -> (B, 3, T, int(F/2))
        x = self.enc(x)

        for i in range(self.num_tsblocks):
            x = self.ts_conformer[i](x)

        # (B, 3, T, int(F/2)) -> (B, num_spk, T, F)
        # (B, num_spk, T, F) -> [(B, T, F), ...]
        masks = [
            m.squeeze(dim=1)
            for m in torch.chunk(self.mask_dec(x), chunks=self._num_spk, dim=1)
        ]
        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        enhanced_mags = [noisy_mag * m for m in masks]
        enhanced_spectrums = [
            mag * torch.exp(1j * noisy_phase) for mag in enhanced_mags
        ]
        # (B, 2*num_spk, T, F) -> Tuple[(B, 2, T, F), ...]
        complex_out = torch.chunk(
            self.complex_dec(x), chunks=self._num_spk, dim=1
        )

        spectrums = []
        for i in range(self._num_spk):
            enhanced_real = enhanced_spectrums[i].real + complex_out[i][
                :, 0, ...
            ].squeeze(dim=1)
            enhanced_imag = enhanced_spectrums[i].imag + complex_out[i][
                :, 1, ...
            ].squeeze(dim=1)
            spectrums.append(torch.complex(enhanced_real, enhanced_imag))

        return spectrums, ilen, others

    @property
    def num_spk(self):
        return self._num_spk
