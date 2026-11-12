"""ASAM: Adaptive Surface representation with Attention and Mamba.

Input: both hemispheres of a subject as patch tensors of shape (B, P, V, C)
(P = 320 Ico-6 patches, V = 153 vertices per patch, C = 3 morphometric features).

Pipeline (Fig. 1 of the paper):
    patch embedding -> split into [Attention | Mamba | CA] streams
    Mamba stream:  local feature learning (DMHS hop search + DFM affinity + gated update),
                   then L/R hemispheres are concatenated with a class token -> Mamba blocks
    CA stream:     channel attention, added to the Attention stream
    Attention stream: L/R concatenated with a class token -> Attention blocks
    prediction = MLP(cls_mamba + cls_attention)
"""
import os
import numpy as np
import torch
import torch.nn as nn
from einops import repeat

from .config import ModelConfig
from .layers import Embeddings, MA, CA, RepFeature, HopProcessing, GatedUpdate, Clf

RESOURCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'ico6')


def load_ico6_neighbourhood(resource_dir=RESOURCE_DIR):
    """Hop neighbourhoods and geodesic distances of the 320 Ico-6 patches.

    hop_patch_ids.npy  (320, 9, 24): ids of the patches at hop j of patch i, padded with -1
    geo_dis_320_float.npy (320, 320): geodesic distance between patch centres
    Padded entries are replaced by the patch itself (distance 0); they are excluded from the
    geodesic affinity because only positive distances are used.
    """
    hop_ids = torch.from_numpy(np.load(os.path.join(resource_dir, 'hop_patch_ids.npy'))).long()
    P = hop_ids.shape[0]
    self_ids = torch.arange(P).view(P, 1, 1).expand_as(hop_ids)
    hop_ids = torch.where(hop_ids == -1, self_ids, hop_ids)
    dis = np.load(os.path.join(resource_dir, 'geo_dis_320_float.npy'))
    dis = torch.from_numpy(dis / np.amax(dis)).float()
    return hop_ids, dis


class ASAM(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig(), resource_dir=RESOURCE_DIR):
        super().__init__()
        self.cfg = cfg
        D, L = cfg.embed_dim, cfg.n_patches
        n_streams = 3 if cfg.use_channel_attention else 2

        # Module names follow the original implementation so that the released checkpoints load directly.
        self.emb_layers = nn.ModuleList([Embeddings(D * n_streams, cfg.in_channels * cfg.n_vertices)])
        self.hop_pro_layers = nn.ModuleList([HopProcessing(D)])
        self.rep_features_layers = nn.ModuleList([
            nn.ModuleList([None] + [RepFeature(D) for _ in range(cfg.k_min, cfg.k_max + 1)])
        ])
        self.ln_gr1 = nn.ModuleList([nn.Linear(D, D)])
        self.agg_gate = nn.ModuleList([GatedUpdate(D)])

        max_len = 2 * L + 1  # left patches + class token + right patches
        self.ma_mam = nn.ModuleList([nn.ModuleList([
            MA(D, cfg.mamba_d_state, cfg.n_heads, max_len, cfg.drop_rate, is_mam=True, is_att=False)
            for _ in range(cfg.mamba_depth)])])
        self.ma_att = nn.ModuleList([nn.ModuleList([
            MA(D, cfg.mamba_d_state, cfg.n_heads, max_len, cfg.drop_rate, is_mam=False, is_att=True)
            for _ in range(cfg.attention_depth)])])
        if cfg.use_channel_attention:
            self.ca = nn.ModuleList([nn.Sequential(*[CA(in_dims=D, seq_len=L) for _ in range(cfg.ca_depth)])])

        self.pos_embedding_mam = nn.ParameterList([nn.Parameter(torch.randn(1, max_len, D))])
        self.pos_embedding_att = nn.ParameterList([nn.Parameter(torch.randn(1, max_len, D))])
        self.cls_token_ma = nn.ParameterList([nn.Parameter(torch.randn(1, 1, D))])
        self.cls_token_att = nn.ParameterList([nn.Parameter(torch.randn(1, 1, D))])
        self.clfs = nn.ModuleList([Clf(D, cfg.n_outputs)])

        hop_ids, geo_dis = load_ico6_neighbourhood(resource_dir)
        self.register_buffer('hop_ids', hop_ids, persistent=False)
        self.register_buffer('geo_dis', geo_dis, persistent=False)

    # ----------------------------------------------------------------------------------
    def hop_distances(self, hop, batch_size):
        """Geodesic distances (B, L, 3*hop) between every patch and its neighbours at ``hop``."""
        L = self.cfg.n_patches
        nbors = self.hop_ids[:, hop, :3 * hop]
        dis = self.geo_dis[torch.arange(L, device=nbors.device).unsqueeze(1), nbors]
        dis = dis.unsqueeze(0).expand(batch_size, -1, -1)
        return dis, torch.ones_like(dis, dtype=torch.bool)

    def fused_affinity(self, x, nbor_feats, nbor_dis):
        """DFM: A^0 = alpha * A_geo + (1 - alpha) * A_dyn, then A = A^0 * 1[A^0 >= epsilon]."""
        cfg = self.cfg
        h_x = self.ln_gr1[0](x)                       # (B, L, D)
        h_n = self.ln_gr1[0](nbor_feats)              # (B, L, n, D)
        A_dyn = torch.sigmoid(torch.einsum('bpd,bpnd->bpn', h_x, h_n))

        valid = nbor_dis[nbor_dis > 0]
        sigma = torch.median(valid) if valid.numel() > 0 else torch.tensor(1.0, device=nbor_dis.device)
        A_geo = torch.exp(-(nbor_dis ** 2) / (sigma ** 2)) * (nbor_dis > 0).float()

        A = cfg.alpha * A_geo + (1 - cfg.alpha) * A_dyn
        A[A < cfg.epsilon] = 0
        return A

    def local_feature_learning(self, x_mam):
        """DMHS: grow the receptive field hop by hop until the survival probability drops below
        kappa (or K_max is reached); collect the selected neighbours, then aggregate them with
        the DFM affinity and fuse with the gated update."""
        cfg = self.cfg
        B, L, D = x_mam.shape
        dev = x_mam.device
        max_nbors = 3 * cfg.k_max * (cfg.k_max + 1) // 2
        roi_hop = torch.zeros((B, L), device=dev)
        roi_dis = torch.zeros((B, L, max_nbors), device=dev)
        roi_feats = torch.zeros((B, L, max_nbors, D), device=dev)

        h_prev = x_mam
        p_surv = torch.ones((B, L), device=dev)
        n_curr = 0
        for j in range(cfg.k_min, cfg.k_max + 1):
            nbors_ids = self.hop_ids[:, j, :3 * j]
            n_j = nbors_ids.shape[-1]
            geo_dis, mask = self.hop_distances(j, B)
            f_j = self.rep_features_layers[0][j](x_mam[:, nbors_ids, :], geo_dis, mask)

            if not torch.any(p_surv >= cfg.kappa):
                break
            h_prev, p_surv = self.hop_pro_layers[0](h_prev, f_j, p_surv)

            if j == cfg.k_min and torch.any(p_surv < cfg.kappa):
                b, l = (p_surv < cfg.kappa).nonzero(as_tuple=True)
                roi_hop[b, l] = cfg.k_min
                roi_feats[b, l, :cfg.k_min * 3, :] = x_mam[:, nbors_ids, :][b, l]
                roi_dis[b, l, :cfg.k_min * 3] = geo_dis[b, l, :cfg.k_min * 3]

            cont = p_surv >= cfg.kappa
            if torch.any(cont):
                b, l = cont.nonzero(as_tuple=True)
                roi_dis[b, l, n_curr:n_curr + n_j] = geo_dis[b, l]
                roi_feats[b, l, n_curr:n_curr + n_j, :] = x_mam[:, nbors_ids, :][b, l]
                n_curr += n_j

            if j > cfg.k_min:
                stop = (p_surv < cfg.kappa) & (roi_hop == 0)
                roi_hop[stop] = cfg.k_min if (j - 1) <= cfg.k_min else j - 1
            if j == cfg.k_max:
                roi_hop[(p_surv >= cfg.kappa) & (roi_hop == 0)] = cfg.k_max

        A = self.fused_affinity(x_mam, roi_feats, roi_dis)         # (B, L, n)
        x_gr = x_mam + torch.einsum('bpn,bpnd->bpd', A, roi_feats)  # message passing with self loop
        return x_mam, x_gr, roi_hop

    @staticmethod
    def _merge_hemispheres(x, cls_token, pos_embedding, B0):
        """(2*B0, L, D) with hemispheres stacked on the batch axis -> (B0, 2L+1, D): [L | cls | R]."""
        L = x.shape[1]
        cls = repeat(cls_token, '() n d -> b n d', b=B0)
        x = torch.cat([x[:B0], cls, x[B0:]], dim=1)
        return x + pos_embedding[:, :2 * L + 1]

    def forward(self, left, right):
        """left, right: (B, P, V, C) patch features of the left / right hemisphere."""
        cfg = self.cfg
        B0, L = left.shape[:2]
        D = cfg.embed_dim
        x = torch.cat([left, right], dim=0)                 # hemispheres share all weights up to fusion
        if self.training and cfg.train_noise_std > 0:
            x = x + cfg.train_noise_std * torch.randn_like(x)
        x = self.emb_layers[0](x)

        if cfg.use_channel_attention:
            x_att, x_mam, ca_feat = torch.split(x, [D, D, D], dim=-1)
            ca_feat = self.ca[0](ca_feat)
        else:
            x_att, x_mam = torch.split(x, [D, D], dim=-1)

        # Mamba stream: local feature learning, then gated fusion with the neighbourhood message
        x_mam, x_gr, _ = self.local_feature_learning(x_mam)
        x_mam = self.agg_gate[0](x_mam, x_gr)
        x_mam = self._merge_hemispheres(x_mam, self.cls_token_ma[0], self.pos_embedding_mam[0], B0)

        # Attention stream (+ channel attention)
        if cfg.use_channel_attention:
            x_att = x_att + ca_feat
        x_att = self._merge_hemispheres(x_att, self.cls_token_att[0], self.pos_embedding_att[0], B0)

        for blk in self.ma_mam[0]:
            x_mam = blk(x_mam)
        for blk in self.ma_att[0]:
            x_att = blk(x_att)

        feat = x_mam[:, L, :] + x_att[:, L, :]             # class tokens of both streams
        return self.clfs[0](feat)
