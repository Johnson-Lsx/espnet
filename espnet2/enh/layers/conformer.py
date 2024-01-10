import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum


class FFN_Module(nn.Module):
    def __init__(
        self, in_feats: int, expansion_factor: int = 4, drop_p: float = 0.1
    ):
        super(FFN_Module, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_feats),
            nn.Linear(in_feats, expansion_factor * in_feats),
            nn.SiLU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion_factor * in_feats, in_feats),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        return self.ffn(x)


class Conformer_Conv_Module(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expansion_factor: int = 2,
        kernel_size: int = 31,
        drop_p: float = 0.0,
    ):
        super(Conformer_Conv_Module, self).__init__()
        inner_chans = expansion_factor * in_channels
        padding = (kernel_size - 1) // 2
        # input shape (B, T, C)
        self.conv = nn.Sequential(
            # (B, T, C)
            nn.LayerNorm(in_channels),
            # (B, C, T)
            Rearrange("b t c -> b c t"),
            # (B, 2 * expansion_factor * C, T)
            nn.Conv1d(in_channels, 2 * inner_chans, 1),
            # (B, expansion_factor * C, T)
            nn.GLU(dim=1),
            nn.Conv1d(
                inner_chans,
                inner_chans,
                kernel_size,
                padding=padding,
                groups=inner_chans,
            ),
            nn.BatchNorm1d(inner_chans),
            nn.SiLU(),
            # (B, C, T)
            nn.Conv1d(inner_chans, in_channels, 1),
            # (B, T, C)
            Rearrange("b c t -> b t c"),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        return self.conv(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_heads: int = 8,
        h_dim: int = 64,
        max_pos_emb: int = 512,
        drop_p: float = 0.0,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        out_dim = num_heads * h_dim

        self.num_heads = num_heads
        self.scale = h_dim**-0.5
        self.max_pos_emb = max_pos_emb

        self.linear_q = nn.Linear(in_dim, out_dim)
        self.linear_kv = nn.Linear(in_dim, 2 * out_dim)
        self.linear_out = nn.Linear(out_dim, in_dim)
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, h_dim)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, mask=None, contex_mask=None):
        # x shape (B, T, D)
        device = x.device
        query_len = x.shape[-2]
        # q, k, v shape (B, T, out_dim)
        q = self.linear_q(x)
        k, v = self.linear_kv(x).chunk(2, dim=-1)
        # reshape q, k, v to (B, num_heads, T, h_dim)
        q, k, v = map(
            lambda t: rearrange(t, "b t (h d) -> b h t d", h=self.num_heads),
            (q, k, v),
        )
        # calculate dot-product attention
        dots = einsum("bhnd,bhmd->bhnm", q, k) * self.scale

        # shaw's relative positional embedding
        # construct the rel_pos matrix
        seq = torch.arange(query_len, device=device)
        # dist shape (query_len, query_len), range (-query_len + 1, query_len - 1)
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
        # convert dist to index matrix for rel_pos_emb table, range (0, 2 * max_pos_emb)
        dist = (
            dist.clamp(-self.max_pos_emb, self.max_pos_emb) + self.max_pos_emb
        )
        # rel_pos shape (query_len, query_len, h_dim)
        rel_pos = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum("bhnd,nrd->bhnr", q, rel_pos) * self.scale
        dots = dots + pos_attn

        # apply mask
        if mask is not None or contex_mask is not None:
            if mask is None:
                mask = torch.ones(*x.size[:2], device=device)
            if contex_mask is None:
                contex_mask = torch.ones(*x.size[:2], device=device)
            # construct the final mask, shape (B, 1, T, T)
            mask = rearrange(mask, "b i - > b () i ()") * rearrange(
                mask, "b j - > b () () j"
            )
            mask_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~mask, mask_value)

        # calculate the attention score
        attn = torch.softmax(dots, dim=-1)
        out = einsum("bhnm,bhmd->bhnd", attn, v)
        out = rearrange(out, "b h n m -> b n (h m)")
        out = self.linear_out(out)
        return self.dropout(out)


class ConformerBlock(nn.Module):
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
        super(ConformerBlock, self).__init__()
        self.ff1 = FFN_Module(dim, ff_expansion_factor, ff_drop)
        self.pre_norm = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads, h_dim, max_pos_emb, attn_drop
        )
        self.conv = Conformer_Conv_Module(
            dim, conv_expansion_factor, kernel_size, conv_drop
        )
        self.ff2 = FFN_Module(dim, ff_expansion_factor, ff_drop)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None, contex_mask=None):
        # x shape (B, T, D)
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(self.pre_norm(x), mask, contex_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.post_norm(x)
