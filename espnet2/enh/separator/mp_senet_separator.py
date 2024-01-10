from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from espnet2.enh.layers.mp_senet import (
    DenseEncoder,
    Mag_Mask_Decoder,
    Phase_Decoder,
    TSConformerBlock,
)
from espnet2.enh.separator.abs_separator import AbsSeparator

EPS = torch.finfo(torch.double).eps


class MP_SENetSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 1,
        in_channels: int = 2,
        hid_chans: int = 64,
        num_tsblocks: int = 4,
        cmp_factor: float = 0.3,
    ):
        """MP-SENet separator

        Reference:
            MP-SENet: A Speech Enhancement Model with Parallel Denoising
            of Magnitude and Phase Spectra
            https://arxiv.org/pdf/2305.13686.pdf

            Audio samples and source codes of the MP-SENet are available at
            https://github.com/yxlu-0102/MP-SENet.

        Args:
            input_dim (int): input feature dimension
            num_spk (int, optional): the number of speakers, currently only support single speaker. Defaults to 1.
            in_channels (int, optional): the number of channels of the input feature. Defaults to 2,
                                         i.e. torch.stack((magnitude, phase), dim=1).
            hid_chans (int, optional): the number of channels of the output of the encoder. Defaults to 64.
            num_tsblocks (int, optional): the number of the TSConformer blocks. Defaults to 4.
            cmp_factor (float, optional): the power compression factor used to compress the magnitude. Defaults to 0.3.
        """
        super(MP_SENetSeparator, self).__init__()
        self._num_spk = num_spk
        self.num_tsblocks = num_tsblocks
        self.cmp_factor = cmp_factor

        self.enc = DenseEncoder(
            in_channels,
            hid_chans,
            hid_chans,
        )

        self.ts_conformer = nn.ModuleList([])
        for i in range(num_tsblocks):
            self.ts_conformer.append(TSConformerBlock(hid_chans))

        self.mag_mask_dec = Mag_Mask_Decoder(
            hid_chans, num_spk, hid_chans, n_freq=input_dim
        )
        self.phase_dec = Phase_Decoder(hid_chans, num_spk, hid_chans)

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            input (torch.Tensor): Encoded complex feature [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch,]
            additional (Optional[Dict], optional): Not used in this model. Defaults to None.

        Returns:
            masked (List[torch.Tensor]): [(Batch, T, F), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        # input shape: (B, T, F)
        noisy_mag = torch.abs(input)
        noisy_phase = torch.angle(input)
        # apply magnitude compression
        cmp_noisy_mag = torch.pow(noisy_mag, self.cmp_factor)
        # input shape: (B, 2, T, F)
        feat = torch.stack((cmp_noisy_mag, noisy_phase), dim=1)
        feat = self.enc(feat)
        for i in range(self.num_tsblocks):
            feat = self.ts_conformer[i](feat)
        # estimate the magnitudes
        # cmp_mag_mask shape: (B, num_spk, T, F)
        cmp_mag_mask = self.mag_mask_dec(feat)
        # cmp_mask_list: List[(B, T, F) x num_spk]
        cmp_mask_list = [
            m.squeeze(dim=1)
            for m in torch.chunk(cmp_mag_mask, self.num_spk, dim=1)
        ]
        others = OrderedDict(
            zip(
                ["mask_spk{}".format(i + 1) for i in range(len(cmp_mask_list))],
                cmp_mask_list,
            )
        )
        cmp_enhanced_mags = [cmp_noisy_mag * m for m in cmp_mask_list]
        enhanced_mags = [
            torch.pow(mag, 1 / self.cmp_factor) for mag in cmp_enhanced_mags
        ]
        # estimate the phases
        # enhanced_phases: List[(B, T, F) x num_spk]
        enhanced_phases = self.phase_dec(feat)
        enhanced_phases = [
            p.squeeze(dim=1)
            for p in torch.chunk(enhanced_phases, self.num_spk, dim=1)
        ]
        # construct the enhanced spectrum
        enhanced_specs = [
            enhanced_mags[i] * torch.exp(1j * enhanced_phases[i])
            for i in range(len(enhanced_mags))
        ]

        return enhanced_specs, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
