from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from packaging.version import parse as V

from espnet2.enh.layers.fullsubnet_plus import (
    FullbandExtractor,
    MulCA,
    SubBandModule,
)
from espnet2.enh.separator.abs_separator import AbsSeparator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class FullSubNet_PlusSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim,
        fb_hid_chans: int = 384,
        num_spk: int = 1,
        s_ks: int = 3,
        m_ks: int = 5,
        l_ks: int = 10,
        M: int = 2,
        N: int = 4,
        ks: int = 3,
        dilations: List[int] = [1, 2, 5, 9],
        sub_out: int = 2,
        sub_nframe: int = 2,
        hidden_size: int = 384,
        num_layers: int = 2,
        bidirectional=False,
        fb_nonlinear: str = "relu",
        sub_nonlinear: str = "tanh",
    ):
        super(FullSubNet_PlusSeparator, self).__init__()

        self._num_spk = num_spk
        self.sub_frame = sub_nframe

        # magnitude branch
        self.mag_mulca = MulCA(input_dim, s_ks, m_ks, l_ks)
        self.mag_fbe = FullbandExtractor(
            input_dim,
            fb_hid_chans,
            fb_hid_chans,
            M,
            N,
            ks,
            dilations,
            fb_nonlinear,
        )
        # real branch
        self.real_mulca = MulCA(input_dim, s_ks, m_ks, l_ks)
        self.real_fbe = FullbandExtractor(
            input_dim,
            fb_hid_chans,
            fb_hid_chans,
            M,
            N,
            ks,
            dilations,
            fb_nonlinear,
        )
        # imaginary branch
        self.imag_mulca = MulCA(input_dim, s_ks, m_ks, l_ks)
        self.imag_fbe = FullbandExtractor(
            input_dim,
            fb_hid_chans,
            fb_hid_chans,
            M,
            N,
            ks,
            dilations,
            fb_nonlinear,
        )
        # sub-band model
        sub_in_feat = 2 * sub_nframe + 1 + 3
        sub_out = sub_out * num_spk
        self.sub_model = SubBandModule(
            sub_in_feat,
            sub_out,
            hidden_size,
            num_layers,
            bidirectional,
            sub_nonlinear,
        )

    def norm(self, input, mode: str = "fb", eps: float = 1e-5):
        assert mode in ["fb", "sb"], f"Not support {mode} mode"
        if mode == "fb":
            # input shape: [B, T, F]
            mu = torch.mean(input, dim=(1, 2), keepdim=True)
        else:
            # input shape: [B, F, T, Fs]
            mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        input = input / (mu + eps)

        return input

    def unfold(self, input):
        # input shape [B, T, F] -> [B, 1, T, F]
        B, T, F = input.shape
        input = input.unsqueeze(dim=1)
        # [B, 1, T , F + 2*sub_frame]
        input = nn.functional.pad(
            input, (self.sub_frame, self.sub_frame, 0, 0), mode="reflect"
        )
        # perform unfold
        input = nn.functional.unfold(
            input, kernel_size=(T, 2 * self.sub_frame + 1)
        )
        # reshape and permute [B, F, T, sub_frame]
        input = input.reshape(B, T, -1, F).permute(0, 3, 1, 2).contiguous()

        return input

    def apply_masks(
        self,
        masks: List[torch.Tensor],
        real: torch.Tensor,
        imag: torch.Tensor,
    ):
        """apply masks

        Args:
            masks : est_masks, [(B, T, F), ...]
            real (torch.Tensor): real part of the noisy spectrum, (B, T, F)
            imag (torch.Tensor): imag part of the noisy spectrum, (B, T, F)

        Returns:
            masked (List[torch.Tensor]): [(B, T, F), ...]
        """
        masked = []
        for i in range(len(masks)):
            # (B, T, F)
            mask_real = masks[i].real
            mask_imag = masks[i].imag
            # (B, T, F)
            real, imag = (
                real * mask_real - imag * mask_imag,
                real * mask_imag + imag * mask_real,
            )
            masked.append(torch.complex(real, imag))
        return masked

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ):
        """_summary_

        Args:
            input (torch.Tensor): Encoded complex feature [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch,]
            additional (Optional[Dict], optional): Not used in this model. Defaults to None.

        Returns:
            masked (List[torch.Tensor]): [(Batch, T, F), ...]
            ilens (torch.Tensor): input lengths [Batch,]
            other predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        # get noisy magnitude, real and imaginary part
        B, T, F = input.shape
        noisy_mag = self.norm(abs(input))
        noisy_real = self.norm(input.real)
        noisy_imag = self.norm(input.imag)

        # magnitude branch forward
        # x_m, psi_m: [B, T, F]
        x_m = self.mag_mulca(noisy_mag)
        psi_m = self.mag_fbe(x_m)
        # psi_s: [B, F, T, 2 * sub_frame + 1]
        psi_s = self.unfold(x_m)

        # real branch forward
        x_r = self.real_mulca(noisy_real)
        psi_r = self.real_fbe(x_r)

        # imaginary branch forward
        x_i = self.imag_mulca(noisy_imag)
        psi_i = self.imag_fbe(x_i)

        # cat all psis together
        # [B, T, F] -> [B, F, T, 1]
        psi_m = rearrange(psi_m, "b t f -> b f t ()")
        psi_r = rearrange(psi_r, "b t f -> b f t ()")
        psi_i = rearrange(psi_i, "b t f -> b f t ()")
        # [B, F, T, 2 * sub_frame + 4] -> [B*F, T, 2 * sub_frame + 4]
        sub_input = torch.cat((psi_s, psi_m, psi_r, psi_i), dim=-1)
        sub_input = self.norm(sub_input, "sb")
        sub_input = rearrange(sub_input, "b f t fs -> (b f) t fs")

        # sub-band model forward
        # [B*F, T, 2 * num_spk] -> [B, F, T, 2 * num_spk]
        masks = self.sub_model(sub_input)
        masks = rearrange(masks, "(b f) t d -> b t f d", f=F)
        # List[(B, T, F, 2) x num_spk]
        ri_mask_list = torch.chunk(masks, chunks=self._num_spk, dim=-1)
        assert is_torch_1_9_plus, "torch version need to be updated"
        mask_list = [
            torch.complex(ri[..., 0], ri[..., 1]) for ri in ri_mask_list
        ]

        masked = self.apply_masks(mask_list, noisy_real, noisy_imag)

        others = OrderedDict(
            zip(
                ["mask_spk{}".format(i + 1) for i in range(len(mask_list))],
                mask_list,
            )
        )

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
