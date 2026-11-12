"""Building blocks of ASAM: patch embedding, Mamba / Attention blocks, channel attention,
local feature learning (DMHS + DFM) and the gated update."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba


class Embeddings(nn.Module):
    """Linear patch embedding: (B, P, V, C) -> (B, P, out_dims)."""

    def __init__(self, out_dims, n_ins):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_ins, out_dims // 8),
            nn.Linear(out_dims // 8, out_dims),
        )
        self.layer_norm = nn.LayerNorm(out_dims, eps=1e-6)

    def forward(self, x):
        B, P, V, C = x.shape
        x = self.proj(x.reshape(B, P, V * C))
        return self.layer_norm(x)


# --------------------------------------------------------------------------------------
# Global sequence modelling: Mamba block and Attention block
# --------------------------------------------------------------------------------------
class Attention(nn.Module):
    """Multi-head self-attention with its own learnable positional embedding."""

    def __init__(self, max_len, dim, heads=8, dropout=0.0):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)  # kept for checkpoint compatibility (unused)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x):
        B, L, _ = x.shape
        x = x + self.pos_embedding[:, :L, :]
        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.heads), self.to_qkv(x).chunk(3, dim=-1))
        attn = self.dropout(self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale))
        out = rearrange(torch.matmul(attn, v), 'b h l d -> b l (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MA(nn.Module):
    """One block of the Mamba branch (``is_mam=True``) or of the Attention branch (``is_att=True``):
    pre-norm token mixer -> bottleneck projection -> residual -> feed-forward."""

    def __init__(self, d_model, n_state, n_heads, max_len, drop_rate=0.4, is_mam=True, is_att=False):
        super().__init__()
        assert is_mam != is_att, 'a block is either a Mamba block or an Attention block'
        self.is_mam = is_mam
        self.is_att = is_att
        if is_mam:
            self.mamba = nn.ModuleList([Mamba(d_model, n_state)])
            self.norm1 = nn.LayerNorm(d_model)
            self.feed_forward1 = nn.Sequential(
                nn.Linear(d_model, d_model // 16),
                nn.Linear(d_model // 16, d_model),
                nn.Dropout(drop_rate),
            )
        if is_att:
            self.att = nn.ModuleList([Attention(dim=d_model, max_len=max_len, heads=n_heads, dropout=drop_rate)])
            self.norm2 = nn.LayerNorm(d_model)
            self.feed_forward2 = nn.Sequential(
                nn.Linear(d_model, d_model // 16),
                nn.Linear(d_model // 16, d_model),
                nn.Dropout(drop_rate),
            )
        self.norm3 = nn.LayerNorm(d_model)  # kept for checkpoint compatibility (unused)
        self.out_ffn = FeedForward(dim=d_model, hidden_dim=d_model // 16, dropout=drop_rate)
        self.weight_att = nn.Parameter(torch.ones(d_model))
        self.weight_mam = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        if self.is_mam:
            out = self.weight_mam * self.feed_forward1(self.mamba[0](self.norm1(x)))
        else:
            out = self.weight_att * self.feed_forward2(self.att[0](self.norm2(x)))
        out = out + x
        return self.out_ffn(out) + out


# --------------------------------------------------------------------------------------
# Channel attention (CA)
# --------------------------------------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv):
        B, Nq, D = x_q.shape
        Nk = x_kv.shape[1]
        H = self.num_heads
        Q = self.to_q(x_q).reshape(B, Nq, H, D // H).transpose(1, 2)
        K = self.to_k(x_kv).reshape(B, Nk, H, D // H).transpose(1, 2)
        V = self.to_v(x_kv).reshape(B, Nk, H, D // H).transpose(1, 2)
        attn = self.dropout(F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1))
        out = (attn @ V).transpose(1, 2).reshape(B, Nq, D)
        return self.out_proj(out)


class CA(nn.Module):
    """Channel attention: the D feature channels attend to a compressed set of D/32 channels,
    with the L patches acting as the feature dimension."""

    def __init__(self, in_dims, seq_len):
        super().__init__()
        self.ln1 = nn.Linear(in_dims, in_dims // 32)
        self.cross = CrossAttention(dim=seq_len, num_heads=1)
        self.layer_norm = nn.LayerNorm(seq_len, eps=1e-6)

    def forward(self, x):                                     # x: (B, L, D)
        x_inter = self.layer_norm(self.ln1(x).transpose(-1, -2))  # (B, D/32, L)
        x = self.layer_norm(x.transpose(-1, -2))                  # (B, D, L)
        return self.cross(x, x_inter).transpose(-1, -2)           # (B, L, D)


# --------------------------------------------------------------------------------------
# Local feature learning: DMHS (hop-wise aggregation + halting) and DFM gating
# --------------------------------------------------------------------------------------
class RepFeature(nn.Module):
    """Distance-weighted aggregation of the neighbours of one hop (Eq. for f_j)."""

    def __init__(self, feature_dim):
        super().__init__()
        self.C = nn.Linear(feature_dim, feature_dim)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, features, distances, mask):
        # features: (B, L, n, D); distances, mask: (B, L, n)
        w = torch.exp(-self.gamma * distances) * mask
        w = w / (w.sum(dim=2, keepdim=True) + 1e-8)
        return self.C(torch.einsum('blnd,bln->bld', features, w))


class HopProcessing(nn.Module):
    """GRU state update and halting probability of the dynamic multi-hop search (DMHS)."""

    def __init__(self, feature_dim):
        super().__init__()
        self.gru = nn.GRUCell(feature_dim, feature_dim)
        self.W_s = nn.Parameter(torch.randn(feature_dim + 1, 1))
        self.b_s = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_prev, f_j, p_surv_prev):
        B, L, D = h_prev.shape
        h_j = self.gru(f_j.reshape(B * L, D), h_prev.reshape(B * L, D)).view(B, L, D)
        delta_h = torch.norm(h_j - h_prev, dim=2)
        s_j = torch.sigmoid(torch.matmul(torch.cat([h_j, delta_h.unsqueeze(2)], dim=2), self.W_s).squeeze(2) + self.b_s)
        return h_j, p_surv_prev * s_j


class GatedUpdate(nn.Module):
    """GRU-style gate that fuses a patch feature h with its aggregated neighbourhood message m."""

    def __init__(self, feature_dim):
        super().__init__()
        self.W_z = nn.Linear(feature_dim, feature_dim)
        self.U_z = nn.Linear(feature_dim, feature_dim)
        self.W_r = nn.Linear(feature_dim, feature_dim)
        self.U_r = nn.Linear(feature_dim, feature_dim)
        self.W_h = nn.Linear(feature_dim, feature_dim)
        self.U_h = nn.Linear(feature_dim, feature_dim)

    def forward(self, h, m):
        z = torch.sigmoid(self.W_z(h) + self.U_z(m))
        r = torch.sigmoid(self.W_r(h) + self.U_r(m))
        h_ = torch.tanh(self.W_h(r * h) + self.U_h(m))
        return (1 - z) * h + z * h_


class Clf(nn.Module):
    def __init__(self, in_dims, n_outputs):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, 64)
        self.fc2 = nn.Linear(64, n_outputs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.sigmoid(self.fc2(self.relu(self.fc1(x))))
