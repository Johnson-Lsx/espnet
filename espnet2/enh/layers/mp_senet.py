import torch
import torch.nn as nn
from einops import rearrange

from espnet2.enh.layers.conformer import ConformerBlock


def get_padding_2D(kernel_size, dilation):
    """calculate the padding that keep the input's shape of Conv2D 
    $$
        \begin{aligned}
            & H_{\text {out }}=\left\lfloor\frac{H_{\text {in }}+2 \times 
                \text { padding }[0]-\operatorname{dilation}[0] \times
                (\text { kernel_size }[0]-1)-1}{\text { stride }[0]}+1\right\rfloor \\
            & W_{\text {out }}=\left\lfloor\frac{W_{\text {in }}+2 \times 
                \text { padding }[1]-\operatorname{dilation}[1] \times
                (\text { kernel_size }[1]-1)-1}{\text { stride }[1]}+1\right\rfloor
        \end{aligned}
    $$

    Note this function assumes the stride for Conv2D is (1, 1)

    Args:
        kernel_size (tuple): the kernel size for Conv2D
        dilation (tuple): the dilation for Conv2D
    """
    padding = (
        int(dilation[0] * (kernel_size[0] - 1) / 2),
        int(dilation[1] * (kernel_size[1] - 1) / 2),
    )
    return padding


class Learnable_Sigmoid(nn.Module):
    def __init__(self, num_feats: int, beta: float = 2.0):
        super(Learnable_Sigmoid, self).__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(num_feats), requires_grad=True)

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_chans: int = 32,
        ksz: tuple = (3, 3),
        depth: int = 4,
    ):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.dense_blocks = nn.ModuleList([])
        for i in range(depth - 1):
            dil = (2**i, 1)
            dense_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels + i * hid_chans,
                    hid_chans,
                    kernel_size=ksz,
                    padding=get_padding_2D(ksz, dil),
                    dilation=dil,
                ),
                nn.InstanceNorm2d(hid_chans, affine=True),
                nn.PReLU(hid_chans),
            )
            self.dense_blocks.append(dense_conv)
        last_conv = nn.Sequential(
            nn.Conv2d(
                in_channels + (depth - 1) * hid_chans,
                out_channels,
                kernel_size=ksz,
                padding=get_padding_2D(ksz, (2 ** (depth - 1), 1)),
                dilation=(2 ** (depth - 1), 1),
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(out_channels),
        )
        self.dense_blocks.append(last_conv)

    def forward(self, input):
        # input shape (B, in_channels, T, F)
        outs = [input]
        for layer in self.dense_blocks:
            x = torch.cat(outs, dim=1)
            x = layer(x)
            outs.append(x)
        # output shape (B, out_channels, T, F)
        return outs[-1]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksz: tuple = (3, 3),
        stride: tuple = (1, 1),
    ) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=ksz,
                stride=stride,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class DeConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksz: tuple = (3, 3),
        stride: tuple = (1, 1),
    ) -> None:
        super(DeConvBlock, self).__init__()
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=ksz,
                stride=stride,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        return self.de_conv(x)


class DenseEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_chans: int,
        ksz: tuple = (1, 3),
        stride: tuple = (1, 2),
        dense_depth: int = 4,
    ):
        super(DenseEncoder, self).__init__()
        self.conv1 = ConvBlock(in_channels, hid_chans, (1, 1))
        self.dense = DenseBlock(
            hid_chans,
            hid_chans,
            hid_chans,
            depth=dense_depth,
        )
        self.conv2 = ConvBlock(
            hid_chans,
            out_channels,
            ksz,
            stride=stride,
        )

    def forward(self, x):
        # x shape (B, in_chans, T, F)
        x = self.conv1(x)
        # x shape (B, hid_chans, T, F)
        x = self.dense(x)
        # x shape (B, out_chans, T, F)
        x = self.conv2(x)
        # x shape (B, out_chans, T, int(F/2))
        return x


class Mag_Mask_Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_chans: int = 32,
        ksz: tuple = (1, 3),
        stride: tuple = (1, 2),
        dense_depth: int = 4,
        n_freq=257,
    ):
        super(Mag_Mask_Decoder, self).__init__()
        self.dense = DenseBlock(
            in_channels, in_channels, in_channels, depth=dense_depth
        )
        self.de_conv = DeConvBlock(in_channels, hid_chans, ksz, stride)
        self.conv2d = nn.Conv2d(hid_chans, out_channels, (1, 1))
        self.lsigmoid = Learnable_Sigmoid(n_freq)

    def forward(self, x):
        x = self.dense(x)
        x = self.de_conv(x)
        x = self.conv2d(x)
        x = self.lsigmoid(x)
        return x


class Phase_Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_chans: int = 32,
        ksz: tuple = (1, 3),
        stride: tuple = (1, 2),
        dense_depth: int = 4,
    ):
        super(Phase_Decoder, self).__init__()
        self.dense = DenseBlock(
            in_channels, in_channels, in_channels, depth=dense_depth
        )
        self.de_conv = DeConvBlock(in_channels, hid_chans, ksz, stride)
        self.r_conv = nn.Conv2d(hid_chans, out_channels, (1, 1))
        self.i_conv = nn.Conv2d(hid_chans, out_channels, (1, 1))

    def forward(self, x):
        x = self.dense(x)
        x = self.de_conv(x)
        phase_real = self.r_conv(x)
        phase_imag = self.i_conv(x)
        phase = torch.atan2(phase_real, phase_imag)
        return phase


class TSConformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        kernel_size: int = 31,
        h_dim: int = 64,
        max_pos_emb: int = 512,
        ff_drop: float = 0.0,
        conv_drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super(TSConformerBlock, self).__init__()
        self.time_conformer = ConformerBlock(
            dim,
            num_heads,
            ff_expansion_factor,
            conv_expansion_factor,
            kernel_size,
            h_dim,
            max_pos_emb,
            ff_drop,
            conv_drop,
            attn_drop,
        )
        self.freq_conformer = ConformerBlock(
            dim,
            num_heads,
            ff_expansion_factor,
            conv_expansion_factor,
            kernel_size,
            h_dim,
            max_pos_emb,
            ff_drop,
            conv_drop,
            attn_drop,
        )

    def forward(self, x):
        # reshape x: (B, C, T, F) -> (B * F, T, C)
        _, _, T, F = x.shape
        x = rearrange(x, "b c t f -> (b f) t c")
        x = x + self.time_conformer(x)
        # reshape x: (B * F, T, C) -> (B * T, F, C)
        x = rearrange(x, "(b f) t c -> (b t) f c", f=F)
        x = x + self.freq_conformer(x)
        # reshape x: (B * T, F, C) -> (B, C, T, F)
        x = rearrange(x, "(b t) f c -> b c t f", t=T)
        return x
