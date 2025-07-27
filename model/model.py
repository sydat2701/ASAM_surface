import os
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import configs.config as cf
from model.MA.MA import MA, FeedForward
from einops import repeat
from einops.layers.torch import Rearrange

class Embeddings(nn.Module):
    def __init__(self, out_dims, n_ins = 3):
        super(Embeddings, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_ins, out_dims//8),
            nn.Linear(out_dims//8, out_dims)
        )
        self.layer_norm = nn.LayerNorm(out_dims, eps=1e-6)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        
        B, P, V, D = x.shape
        x = x.view(B, P, -1)
        x = self.proj(x)
        x = self.layer_norm(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5  # Scaling factor for dot-product attention
        
        # Linear layers for Q, K, V
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_q, x_kv, mask=None):
        """
        x_q: Query tensor of shape (B, Nq, D)
        x_kv: Key/Value tensor of shape (B, Nk, D)
        mask: Optional mask of shape (B, 1, Nq, Nk)
        """
        B, Nq, D = x_q.shape
        Nk = x_kv.shape[1]
        H = self.num_heads
        
        # Compute Q, K, V
        Q = self.to_q(x_q).reshape(B, Nq, H, D // H).transpose(1, 2)  # (B, H, Nq, D/H)
        K = self.to_k(x_kv).reshape(B, Nk, H, D // H).transpose(1, 2)  # (B, H, Nk, D/H)
        V = self.to_v(x_kv).reshape(B, Nk, H, D // H).transpose(1, 2)  # (B, H, Nk, D/H)
        
        # Compute attention scores
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nk)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)  # Softmax over Nk
        attn_probs = self.dropout(attn_probs)
        
        # Compute attention output
        out = (attn_probs @ V).transpose(1, 2).reshape(B, Nq, D)  # (B, Nq, D)
        out = self.out_proj(out)
        return out 
    
class CA(nn.Module):
    def __init__(self, in_dims, seq_len):
        super(CA, self).__init__()
        self.ln1 = nn.Linear(in_dims, in_dims//32)
        self.cross = CrossAttention(dim = seq_len, num_heads=1)
        self.layer_norm = nn.LayerNorm(seq_len, eps=1e-6)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        res = x
        B, L, D = x.shape

        x_inter = self.ln1(x) #B, L, d
        x_inter = x_inter.transpose(-1, -2)
        x_inter = self.layer_norm(x_inter)
        x = self.layer_norm(x.transpose(-1,-2))

        cross_att = self.cross(x, x_inter) #B, D, L
        x = cross_att.transpose(-1,-2)
        #x = x + res
        return x

class Clf(nn.Module):
    def __init__(self, in_dims, n_classes):
        super(Clf, self).__init__()
        self.fc1 = nn.Linear(in_dims, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x


class RepFeature(nn.Module):
    def __init__(self, feature_dim):
        super(RepFeature, self).__init__()

        self.C = nn.Linear(feature_dim, feature_dim)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, features, distances, mask):
        """
        Args:
            features: Tensor of shape (B, L, n_nbors, D) - Neighbor features.
            distances: Tensor of shape (B, L, n_nbors) - Geodesic distances.
            mask: Tensor of shape (B, L, n_nbors) - Boolean mask (1 for valid neighbors, 0 for padding).
        Returns:
            aggregated_features: Tensor of shape (B, L, D) - Aggregated features for all hops.
        """
        # Compute distance-based weights
        distance_weights = torch.exp(-self.gamma * distances)  # Shape: (B, L, n_nbors)

        # Apply mask: Zero out weights for padded neighbors
        distance_weights = distance_weights * mask  # Shape: (B, L, n_nbors)

        # Normalize weights: Prevent division by zero (only sum over valid neighbors)
        distance_weights = distance_weights / (distance_weights.sum(dim=2, keepdim=True) + 1e-8)  

        # Compute weighted aggregation (vectorized)
        weighted_features = torch.einsum('blnd,bln->bld', features, distance_weights)  # (B, L, D)

        # Apply learnable projection
        aggregated_features = self.C(weighted_features)  # (B, L, D)
        return aggregated_features
    
    
class HopProcessing(nn.Module):
    def __init__(self, feature_dim):
        super(HopProcessing, self).__init__()
        self.feature_dim = feature_dim

        # GRU for state update (2b)
        self.gru = nn.GRUCell(feature_dim, feature_dim)

        # Learnable parameters for halting probability (2d)
        self.W_s = nn.Parameter(torch.randn(feature_dim + 1, 1))  # +1 for delta_h
        self.b_s = nn.Parameter(torch.tensor(0.0))  # Bias

    def forward(self, h_prev, f_j, p_surv_prev):
        """
        Args:
            h_prev: Tensor of shape (B, L, D) - Hidden state from the previous hop.
            f_j: Tensor of shape (B, L, D) - Aggregated feature for the current hop.
            p_surv_prev: Tensor of shape (B, L) - Cumulative survival probability up to the previous hop.
        Returns:
            h_j: Tensor of shape (B, L, D) - Updated hidden state.
            p_surv: Tensor of shape (B, L) - Survival probability for the current hop.
        """
        B, L, D = h_prev.shape

        # 2b: State Update using GRU (process each location in L independently)
        h_j = torch.zeros_like(h_prev)  # Initialize output tensor
        for i in range(L):
            h_j[:, i, :] = self.gru(f_j[:, i, :], h_prev[:, i, :])  # Apply GRU per location

        # 2c: State Change Measurement
        delta_h = torch.norm(h_j - h_prev, dim=2)  # L2 norm, shape: (B, L)

        # 2d: Halting Probability
        s_j_input = torch.cat([h_j, delta_h.unsqueeze(2)], dim=2)  # Shape: (B, L, D + 1)
        s_j = torch.sigmoid(torch.matmul(s_j_input, self.W_s).squeeze(2) + self.b_s)  # Shape: (B, L)

        # 2e: Stopping Probability
        p_surv = p_surv_prev * s_j  # Shape: (B, L)

        return h_j, p_surv
    

class GatedUpdate(nn.Module):
    def __init__(self, feature_dim):
        super(GatedUpdate, self).__init__()
        
        self.W_z = nn.Linear(feature_dim, feature_dim)
        self.U_z = nn.Linear(feature_dim, feature_dim)
        
        self.W_r = nn.Linear(feature_dim, feature_dim)
        self.U_r = nn.Linear(feature_dim, feature_dim)
        
        self.W_h = nn.Linear(feature_dim, feature_dim)
        self.U_h = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, h, m):
        """
        h: (B, L, D)  -> Current node features
        m: (B, L, D)  -> Aggregated messages from neighbors
        Returns:
        h_new: (B, L, D) -> Updated node features
        """
        z = torch.sigmoid(self.W_z(h) + self.U_z(m))  # Update gate
        r = torch.sigmoid(self.W_r(h) + self.U_r(m))  # Reset gate
        
        h_ = torch.tanh(self.W_h(r * h) + self.U_h(m))  # Candidate hidden state
        
        h_new = (1 - z) * h + z * h_  # Final node update
        
        return h_new
    

class ASAM(nn.Module):
    def __init__(self, num_res=cf.n_res, n_classes = cf.output_units, is_mam=cf.is_mam, is_att = cf.is_att):
        super(ASAM, self).__init__()

        self.num_res = num_res
        self.rep_features_layers = nn.ModuleList([])
        self.emb_layers = nn.ModuleList([])
        self.hop_pro_layers = nn.ModuleList([])
        self.is_mam = is_mam
        self.is_att = is_att

        for i in range(num_res):
            add = 1 if cf.is_channel_att else 0
            if self.is_att and self.is_mam:
                out_dims = cf.out_emb_dims[i]*(2+add)
            else:
                out_dims = cf.out_emb_dims[i]
            self.emb_layers.append(Embeddings(out_dims, cf.input_dims*cf.n_vertices[i]))

            self.hop_pro_layers.append(HopProcessing(cf.out_emb_dims[i]))

            per_hop = nn.ModuleList([None])
            for _ in range(cf.MIN_HOP[i], cf.MAX_HOP[i]+1):
                per_hop.append(RepFeature(cf.out_emb_dims[i]))
            self.rep_features_layers.append(per_hop)
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        
        self.ln_gr1 = nn.ModuleList([nn.Linear(cf.out_emb_dims[i], cf.out_emb_dims[i]) for i in range(cf.n_res)])

        self.agg_gate = nn.ModuleList([GatedUpdate(cf.out_emb_dims[i]) for i in range(self.num_res)])

        add_ = 1 if cf.is_att and cf.is_cls_token else 0
        self.ma_mam = nn.ModuleList([
            nn.ModuleList([MA(d_model=cf.out_emb_dims[i], n_state=cf.n_states[i], n_heads=cf.n_heads[i], max_len=2*cf.n_patches[i]+add_, \
               num_res=self.num_res, depth=1, is_mam=True, is_att=False) for _ in range(cf.mam_dep)]) for i in range(self.num_res)
        ])
        self.ma_att = nn.ModuleList([
            nn.ModuleList([MA(d_model=cf.out_emb_dims[i], n_state=cf.n_states[i], n_heads=cf.n_heads[i], max_len=2*cf.n_patches[i]+add_, \
               num_res=self.num_res, depth=1, is_mam=False, is_att=True) for _ in range(cf.att_dep)]) for i in range(self.num_res)
        ])

        if self.is_att and self.is_mam:
            self.prj_layers = nn.ModuleList([
                nn.ModuleList([nn.Linear(cf.out_emb_dims[i], 2*cf.out_emb_dims[i]) for _ in range(cf.MA_depths[i]-1)]) for i in range(self.num_res)
            ])

        if cf.is_channel_att:
            if cf.is_att and cf.is_mam:
                factor =2
            else:
                factor=1

            self.ca = nn.ModuleList([
                nn.Sequential(*[
                    CA(in_dims=cf.out_emb_dims[i]*1, seq_len=cf.n_patches[i]) 
                    for _ in range(cf.CA_depth)  # Number of CA layers per resolution
                ])
                for i in range(self.num_res)  # Number of resolutions
            ])

        if cf.is_mam:
            self.pos_embedding_mam = nn.ParameterList([
                nn.Parameter(torch.randn(1, cf.n_patches[i]*2 + 1, cf.out_emb_dims[i])) for i in range(self.num_res)
            ])
        if cf.is_att:
            self.pos_embedding_att = nn.ParameterList([
                nn.Parameter(torch.randn(1, cf.n_patches[i]*2 + 1, cf.out_emb_dims[i])) for i in range(self.num_res)
            ])


        self.K_hop_list = [dict() for i in range(self.num_res)]
        self.K_hop_list_ori = [dict() for i in range(self.num_res)] #with padding -1
        self.geo_dis = []

        self.initialize(self.num_res)

        self.clfs = nn.ModuleList()
        if not cf.is_cls_token:
            for i in range(self.num_res):
                self.clfs.append(Clf(in_dims=2*cf.n_patches[i]*cf.out_emb_dims[i], n_classes=n_classes))
        else:
            self.cls_token_ma = nn.ParameterList([
                nn.Parameter(torch.randn(1, 1, cf.out_emb_dims[i])) for i in range(self.num_res)
            ])
            self.cls_token_att = nn.ParameterList([
                nn.Parameter(torch.randn(1, 1, cf.out_emb_dims[i])) for i in range(self.num_res)
            ])
            for i in range(self.num_res):
                self.clfs.append(Clf(in_dims=cf.out_emb_dims[i], n_classes=n_classes))
        
        
    def initialize(self, num_res):
        
        
        for resol_idx in range(num_res):
            patch_ids = torch.from_numpy(np.load("new_icos/ico6/hop_patch_ids.npy")).long()
            self.K_hop_list_ori[resol_idx] = patch_ids

            P, n_hops, n_nbors = patch_ids.shape
            for i in range(P):
                for j in range(n_hops):
                    tmp = patch_ids[i,j,:]
                    patch_ids[i,j][tmp==-1] = i

            self.K_hop_list[resol_idx] = patch_ids
        
            dis = np.load("new_icos/ico6/geo_dis_320_float.npy")
            dis = dis/np.amax(dis) #normalization
            self.geo_dis.append(torch.from_numpy(dis).float()) #(num patches, num patches)
    
    def geodesic_dis(self, resol_idx, ihop_ids, batch_size):
        #TODO
        '''
            determine the IHOP for ALL patches of resol_idx
            return:
                the mask marks where are the real neighbors (because padded by -1)
        '''
        num_nbors = 3*ihop_ids
        nbors_at_ihop = self.K_hop_list_ori[resol_idx][:, ihop_ids, :num_nbors]
        mask = nbors_at_ihop!=-1

        dis = self.geo_dis[resol_idx][torch.arange(cf.n_patches[resol_idx]).unsqueeze(1), nbors_at_ihop]

        dis = dis.unsqueeze(0).repeat(batch_size, 1, 1) #(B, n_patches, num_nbors)
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)


        return dis.to(self.device), mask.to(self.device)

    def construct_graph(self, X, ith_res, roi_nbor_feats, roi_geo_dis, method='median', alpha = 0.5, THRESH=cf.thresh_range[1]): #-------------------hyper-params---------
        '''
        the number of neighborhood patches in i-hop is 3i (i=0,1,...) => total number of neighborhood patches upto i-hop is 3i*(i+1)/2
        roi_geo_dis was normalized to (0,1] in advanced based on Gaussian kernel

        X: input features of shape (B, L/npatches, D)
        roi_hops: (B, npatches)
        roi_geo_dis: (B, npatches, n_nbors:padded)
        '''

        B, L, D = X.shape
        _, _, max_nbors, _ = roi_nbor_feats.shape

        h_x = self.ln_gr1[ith_res](X) #(B, L, D)
        h_nbrs = self.ln_gr1[ith_res](roi_nbor_feats) #(B, L, nbors, D)

        A_dynamic = torch.einsum("bpd,bpnd->bpn", h_x, h_nbrs) #(B, L, nbors)
        A_dynamic = F.sigmoid(A_dynamic)

        
        valid_distances = roi_geo_dis[roi_geo_dis > 0]  # Ignore padded distances (0)

        if valid_distances.numel() > 0:  # Ensure we have valid distances
            if method == "median":
                sigma = torch.median(valid_distances)
            elif method == "mean":
                sigma = torch.mean(valid_distances) / 3  # More stable scaling
        else:
            sigma = torch.tensor(1.0, device=roi_geo_dis.device)  # Fallback value to prevent NaN

        A_geo = torch.exp(- (roi_geo_dis ** 2) / (sigma ** 2))
        A_geo = A_geo * (roi_geo_dis > 0).float()  # Masking padded values #(B, L, nbors)

        A = alpha*A_geo + (1-alpha)*A_dynamic
        A[A<THRESH] = 0

        return A


    def forward(self, x):
        B, L, _, _ = x.shape
        
        B0 = B//2

        max_nbors = [int(3*cf.MAX_HOP[i]*(cf.MAX_HOP[i]+1)/2) for i in range(cf.n_res)] #maximum number of neighbors (in total, corresponds to max hops) for each resol #(3, (n_max_nbors))
        x_att = None
        x_mam = None

        x = [x]

        for idx, elm in enumerate(x):
            x[idx] = self.emb_layers[idx](x[idx]) #(B, L, D)

            if self.is_mam and self.is_att:
                if cf.is_channel_att:
                    x_att, x_mam, ca_feat= torch.split(x[idx], [cf.out_emb_dims[idx], cf.out_emb_dims[idx], cf.out_emb_dims[idx]], dim=-1)
                else:
                    x_att, x_mam= torch.split(x[idx], [cf.out_emb_dims[idx], cf.out_emb_dims[idx]], dim=-1)

            if cf.is_channel_att:
                ca_feat = self.ca[idx](ca_feat)

            if self.is_att:
                if not self.is_mam:
                    x_att = x[idx]

            if self.is_mam:
                if not self.is_att:
                    x_mam = x[idx]
                roi_hop_per_patch = torch.zeros((B, cf.n_patches[idx])).to(self.device) 
                roi_geo_dis_per_patch = torch.zeros((B, cf.n_patches[idx], max_nbors[idx])).to(self.device)
                roi_nbor_feats = torch.zeros((B, cf.n_patches[idx], max_nbors[idx], cf.out_emb_dims[idx])).to(self.device) #-----------initial-------------

                patch = x_mam

                h_prev = patch #h=f0 #(B, L, D)
                p_surv_prev = torch.ones((B,cf.n_patches[idx])).to(self.device) #(B,L)

                n_curr_nbors = 0
                for j in range(cf.MIN_HOP[idx], cf.MAX_HOP[idx]+1):
                    num_nbors_hop_j = 3*j #this is upper bound
                    nbors_ids = self.K_hop_list[idx][:, j, :][:, :num_nbors_hop_j]
                    n_nbors_of_hop_j = nbors_ids.shape[-1] 
                    geo_dis, mask = self.geodesic_dis(idx, j, B) 

                    jhop_rep = self.rep_features_layers[idx][j](x_mam[:, nbors_ids, :], geo_dis, mask)

                    if torch.any(p_surv_prev>=cf.SURV_HOP_THRES_PROB[idx]): 
                        h_prev, p_surv_prev = self.hop_pro_layers[idx](h_prev, jhop_rep, p_surv_prev) #h_prev: (B, L, D); p_surv_prev = (B,L) 

                        if j==cf.MIN_HOP[idx] and torch.any(p_surv_prev<cf.SURV_HOP_THRES_PROB[idx]):
                            marked = (p_surv_prev<cf.SURV_HOP_THRES_PROB[idx]) #(B, L)
                            roi_hop_per_patch[marked] = cf.MIN_HOP[idx]

                            indices = marked.nonzero(as_tuple=True)
                            roi_nbor_feats[indices[0], indices[1], :cf.MIN_HOP[idx]*3, :] = x_mam[:, nbors_ids, :][indices[0], indices[1]]

                            indices = marked.nonzero(as_tuple=True)
                            roi_geo_dis_per_patch[indices[0], indices[1], :cf.MIN_HOP[idx]*3] = geo_dis[indices[0], indices[1], :cf.MIN_HOP[idx]*3]
                        marked_continue = (p_surv_prev>=cf.SURV_HOP_THRES_PROB[idx]) #(B, L)

                        if True in marked_continue:
                            indices = marked_continue.nonzero(as_tuple=True)
                            roi_geo_dis_per_patch[indices[0], indices[1], n_curr_nbors:n_curr_nbors+n_nbors_of_hop_j] = geo_dis[indices[0], indices[1]]      
                            
                            indices = marked_continue.nonzero(as_tuple=True)
                            roi_nbor_feats[indices[0], indices[1], n_curr_nbors:n_curr_nbors+n_nbors_of_hop_j, :] = x_mam[:, nbors_ids, :][indices[0], indices[1]]
                            n_curr_nbors += n_nbors_of_hop_j

                        if j>cf.MIN_HOP[idx]:
                            marked = (p_surv_prev<cf.SURV_HOP_THRES_PROB[idx]) & (roi_hop_per_patch==0)
                            if True in marked:
                                roi_hop = cf.MIN_HOP[idx] if (j-1)<=cf.MIN_HOP[idx] else j-1
                                roi_hop_per_patch[marked] = roi_hop
                        
                        if j==cf.MAX_HOP[idx]:
                            marked = (p_surv_prev>=cf.SURV_HOP_THRES_PROB[idx]) & (roi_hop_per_patch==0)
                            if True in marked:
                                roi_hop = cf.MAX_HOP[idx]
                                roi_hop_per_patch[marked] = roi_hop
                    else:
                        break
            if self.is_mam:
                roi_hop_per_patch_flat = roi_hop_per_patch.flatten().long()
                num_roi_output = torch.bincount(roi_hop_per_patch_flat, minlength=6)

                A = self.construct_graph(x_mam, idx, roi_nbor_feats, roi_geo_dis_per_patch) #(B, L, n_bors)
        
                x_gr = torch.einsum('bpn,bpnd->bpd', A, roi_nbor_feats)

                x_gr = x_mam + x_gr 
                x_mam = self.agg_gate[idx](x_mam, x_gr) 

                left_hemi = x_mam[:B0, :, :]
                right_hemi = x_mam[B0:, :, :]
                x_mam = torch.cat([left_hemi, right_hemi], dim=1) #(B, 2L, D)

                if cf.is_cls_token:
                    cls_tokens1 = repeat(self.cls_token_ma[idx], '() n d -> b n d', b = B0)
                    x_mam = torch.cat([left_hemi, cls_tokens1, right_hemi], dim=1) #(B, 2L+1, D)
                    x_mam += self.pos_embedding_mam[idx][:, :(2*L + 1)]
            
            if cf.is_att and cf.is_channel_att:
                x_att = x_att + ca_feat
            
            if cf.is_att:
                x_att = torch.cat([x_att[:B0,:,:], x_att[B0:, :, :]], dim=1)

            if cf.is_att and cf.is_cls_token:
                left_hemi, right_hemi = x_att[:, :L, :], x_att[:, L:, :]
                cls_tokens2 = repeat(self.cls_token_att[idx], '() n d -> b n d', b = B0)
                x_att = torch.cat([left_hemi, cls_tokens2, right_hemi], dim=1) #(B, 2L+1, D)
                x_att += self.pos_embedding_att[idx][:, :(2*L + 1)]

            for i in range(cf.mam_dep):
                x_mam = self.ma_mam[idx][i](None, x_mam)

            for i in range(cf.att_dep):
                x_att = self.ma_att[idx][i](x_att, None)

            x[idx] = x_mam[:, L, :] + x_att[:, L, :]  
            x[idx] = self.clfs[idx](x[idx].view(B0, -1))

        return x[idx], num_roi_output
            

        

            









        


