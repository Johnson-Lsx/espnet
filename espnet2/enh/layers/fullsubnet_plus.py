from typing import List

import torch
import torch.nn as nn

EPS = torch.finfo(torch.get_default_dtype()).eps


class GlobalLayerNorm(nn.Module):
    def __init__(self, in_channels, eps: float = 1e-5):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.parameter.Parameter(torch.empty(1, in_channels, 1))
        self.beta = nn.parameter.Parameter(torch.empty(1, in_channels, 1))
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, input):
        # input shape [B, C, L]
        mean = torch.mean(input, dim=(1, 2), keepdim=True)
        var = torch.var(input, dim=(1, 2), unbiased=False, keepdim=True)
        input_norm = (
            self.gamma * ((input - mean) / (torch.sqrt(var + self.eps)))
            + self.beta
        )
        return input_norm


class TCNBlock(nn.Module):
    def __init__(self, in_channels, hid_chans, dd_chans, ks, dilation):
        super(TCNBlock, self).__init__()
        padding = int(dilation * (ks - 1) / 2)

        self.pre_conv = nn.Sequential(
            nn.Conv1d(in_channels, hid_chans, kernel_size=1),
            nn.PReLU(hid_chans),
            GlobalLayerNorm(hid_chans),
        )

        self.dd_conv = nn.Sequential(
            nn.Conv1d(
                hid_chans,
                hid_chans,
                ks,
                padding=padding,
                dilation=dilation,
                groups=hid_chans,
            ),
            nn.Conv1d(hid_chans, dd_chans, 1),
            nn.PReLU(dd_chans),
            GlobalLayerNorm(dd_chans),
        )

        self.post_conv = nn.Conv1d(dd_chans, in_channels, 1)

    def forward(self, input):
        # input shape [B, T, F] -> [B, F, T]
        input_T = input.transpose(-2, -1)
        # [B, hid_chans, T]
        input_T = self.pre_conv(input_T)
        # [B, dd_chans, T]
        input_T = self.dd_conv(input_T)
        # [B, F, T]
        input_T = self.post_conv(input_T)
        # [B, T, F]
        output = input + input_T.transpose(-2, -1)

        return output


class TCNModule(nn.Module):
    def __init__(
        self,
        in_chans,
        hid_chans,
        dd_chans,
        N: int = 4,
        ks: int = 3,
        dilations: List[int] = [1, 2, 5, 9],
    ):
        super(TCNModule, self).__init__()
        assert len(dilations) == N

        self.num_layers = N
        self.tcns = nn.ModuleList([])

        for i in range(N):
            self.tcns.append(
                TCNBlock(in_chans, hid_chans, dd_chans, ks, dilations[i])
            )

    def forward(self, input):
        # input shape [B, T, F]
        for i in range(self.num_layers):
            input = self.tcns[i](input)

        return input


class MulCA(nn.Module):
    def __init__(
        self,
        in_channels: int = 257,
        s_ks: int = 3,
        m_ks: int = 5,
        l_ks: int = 10,
    ):
        """Multi-scale Time Sensitive Channel Attention

        Args:
            in_channels (int, optional): the number of input channels. Defaults to 257.
            s_ks (int, optional): small kernel size of parallel 1-D convolutions. Defaults to 3.
            m_ks (int, optional): medium kernel size of parallel 1-D convolutions. Defaults to 5.
            l_ks (int, optional): large kernel size of parallel 1-D convolutions. Defaults to 10.
        """
        super(MulCA, self).__init__()
        self.s_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, s_ks, groups=in_channels),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True),
        )

        self.m_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, m_ks, groups=in_channels),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True),
        )

        self.l_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, l_ks, groups=in_channels),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True),
        )

        self.fusion_fc = nn.Linear(3, 1)
        self.fcs = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, input):
        # input shape [B, T, F] -> [B, F, T]
        input = input.transpose(-2, -1)

        # Cs, Cm, Cl shape [B, F, 1]
        Cs = self.s_conv(input)
        Cm = self.m_conv(input)
        Cl = self.l_conv(input)
        # [B, F, 3] -> [B, F]
        C_input = torch.cat((Cs, Cm, Cl), dim=-1)
        C_input = self.fusion_fc(C_input).squeeze(dim=-1)
        output = self.fcs(C_input).unsqueeze(dim=-1)

        # [B, F, T] -> [B, T, F]
        output = (input * output).transpose(-2, -1)

        return output


class FullbandExtractor(nn.Module):
    def __init__(
        self,
        in_channels,
        hid_chans,
        dd_chans,
        M: int = 2,
        N: int = 4,
        ks: int = 3,
        dilations: List[int] = [1, 2, 5, 9],
        nonlinear: str = "relu",
    ):
        """Full-band Extractor

        Args:
            M (int, optional): number of groups of TCN blocks. Defaults to 2.
            N (int, optional): number of TCN blocks in each group. Defaults to 4.
            ks (int, optional): kernel size of convolution layers in TCN blocks. Defaults to 3.
            dilations (List[int], optional): dilation rates in each TCN block. Defaults to [1, 2, 5, 9].
            nonlinear (str, Optional): the activation function of Linear layer.
        """
        super(FullbandExtractor, self).__init__()

        assert nonlinear in [
            "prelu",
            "relu",
            "tanh",
            "sigmoid",
        ], f"Not supporting nonlinear={nonlinear}"

        self.num_groups = M
        self.tcn_groups = nn.ModuleList([])

        for i in range(M):
            self.tcn_groups.append(
                TCNModule(in_channels, hid_chans, dd_chans, N, ks, dilations)
            )

        self.fc = nn.Linear(in_channels, in_channels)

        self.active = {
            "prelu": nn.PReLU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }[nonlinear]

    def forward(self, input):
        # input shape [B, T, F]
        for i in range(self.num_groups):
            input = self.tcn_groups[i](input)

        input = self.active(self.fc(input))
        return input


class SubBandModule(nn.Module):
    def __init__(
        self,
        feat_dim,
        out_dim,
        hidden_size: int = 384,
        num_layers: int = 2,
        bidirectional=False,
        nonlinear: str = "relu",
    ):
        super(SubBandModule, self).__init__()

        assert nonlinear in [
            "prelu",
            "relu",
            "tanh",
            "sigmoid",
        ], f"Not supporting nonlinear={nonlinear}"

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        h_out = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(h_out, out_dim)

        self.active = {
            "prelu": nn.PReLU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }[nonlinear]

    def forward(self, input):
        # input shape [B, T, D]
        input, _ = self.lstm(input)
        # [B, T, out_dim]
        input = self.active(self.fc(input))

        return input
