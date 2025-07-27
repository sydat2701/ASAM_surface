import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm import Mamba
import warnings
warnings.filterwarnings('ignore')

class Attention(nn.Module):
    def __init__(self, max_len, dim, heads=8, dropout=0.0):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        )

        # Positional Embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x):
        # x: (B, L, D)

        B, L, _ = x.shape
        x = x + + self.pos_embedding[:, :L, :]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, heads, L, L)
        
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, heads, L, d)
        out = rearrange(out, 'b h l d -> b l (h d)')
        
        return self.to_out(out)

    
class HymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HymbaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MA(nn.Module):
    def __init__(self, d_model, n_state, n_heads, max_len, num_res, depth=2, drop_rate=0.4, is_mam = True, is_att=True):
        super(MA, self).__init__()
        self.d_model = d_model

        self.is_mam = is_mam
        self.is_att = is_att

        if is_mam:
            self.mamba = nn.ModuleList([
                    Mamba(d_model, n_state) for _ in range(depth)
            ])
            
            # Norm and feed-forward network layer
            self.norm1 = nn.LayerNorm(d_model)
            self.feed_forward1 = nn.Sequential(
                nn.Linear(d_model, d_model//16), 
                nn.Linear(d_model//16, d_model),
                nn.Dropout(drop_rate)
            )

        if is_att:
            self.att = nn.ModuleList([
                Attention(dim=d_model, max_len=max_len, heads=n_heads, dropout=drop_rate) for _ in range(depth)
            ])

            self.norm2 = nn.LayerNorm(d_model)
        
            self.feed_forward2 = nn.Sequential(
                nn.Linear(d_model, d_model//16), 
                nn.Linear(d_model//16, d_model),
                nn.Dropout(drop_rate)
            )

        self.norm3 = nn.LayerNorm(d_model)
        self.out_ffn = FeedForward(dim=d_model, hidden_dim=d_model//16, dropout=drop_rate)
        self.weight_att = nn.Parameter(torch.ones(d_model))
        self.weight_mam = nn.Parameter(torch.ones(d_model))
        self.depth = depth

    def forward(self, x_att=None, x_mamba=None):
        '''
            x: B, L, D
        '''

        if self.is_att and self.is_mam:
            res = x_att + x_mamba
        elif not self.is_mam:
            res = x_att
        else:
            res = x_mamba
        
        att_out = 0.
        mamba_out = 0.

        if self.is_mam:
            # Forward Mamba
            x_mamba = self.norm1(x_mamba)
            for i in range(self.depth):
                mamba_out = self.mamba[i](x_mamba)
            mamba_out = self.feed_forward1(mamba_out)

        if self.is_att:
            #forward Att
            x_att = self.norm2(x_att)
            for i in range(self.depth):
                att_out = self.att[i](x_att)
            att_out = self.feed_forward2(att_out)

        ma = self.weight_att * att_out + self.weight_mam * mamba_out
        ma = ma + res
        ma = self.out_ffn(ma) + ma
        

        return ma

# model = MA(d_model=64, n_state=16, n_heads=8, max_len=32).cuda()
# x = torch.randn((4,32,64)).cuda()
# out = model(x, x)
# print(out.shape)














