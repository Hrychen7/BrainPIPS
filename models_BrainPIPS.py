from functools import partial
import torch
import torch.nn as nn
import random
import numpy as np
import math
from timm.utils import ModelEma
from timm.models.layers import DropPath
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from util.pos_embed import get_sinusoid_encoding_table
class BNTF(nn.Module):
    def __init__(self,depth,heads,dim_feedforward):
        super().__init__()
        self.num_patches = 100

        self.attention_list = nn.ModuleList()
        self.node_num = 100
        for _ in range(int(depth)):
            self.attention_list.append(
                TransformerEncoderLayer(d_model=self.node_num, nhead=int(heads), dim_feedforward=dim_feedforward, 
                                        batch_first=True)
            )
       
    def forward(self,img,forward_with_mlp=False):
        bz, _, _, = img.shape

        for atten in self.attention_list:
            img = atten(img)
        
        return img
# Progressive Optimal Transport        
class SemiCurrSinkhornKnopp(nn.Module):
    def __init__(self, num_iters=3, epsilon=0.1, gamma=1, stoperr=1e-6,
                 numItermax=50, rho=1.0, semi_use=True, parcellation_type=7):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.gamma = gamma
        self.stoperr = stoperr
        self.numItermax = numItermax
        self.rho = rho
        self.semi_use = semi_use
        self.parcellation_type = parcellation_type
        self.b = None
        self.register_buffer('cluster_prior', self._build_cluster_prior(parcellation_type))
        
    def _build_cluster_prior(self, parcellation_type):
        if parcellation_type == 7:
            cluster_sizes = [17, 14, 15, 12, 5, 13, 24]
        else:
            return torch.ones(1, 1)
        cluster_sizes = torch.tensor(cluster_sizes, dtype=torch.float64)
        cluster_prior = cluster_sizes / cluster_sizes.sum()
        return cluster_prior.view(-1, 1)

    @torch.no_grad()
    def forward(self, P):
        P = P.detach().double()
        P = -torch.log_softmax(P, dim=1)
       
        n, k = P.shape
        mu = torch.zeros(n, 1, dtype=torch.float64, device=P.device)
        expand_cost = torch.cat([P, mu], dim=1)
        Q = torch.exp(-expand_cost / self.epsilon)
        
        Pa = torch.ones(n, 1, dtype=torch.float64, device=P.device) / n
        Pb = torch.cat([
            self.rho * self.cluster_prior.to(P.device),
            torch.tensor([[1 - self.rho]], dtype=torch.float64, device=P.device)
        ], dim=0)

        b = torch.ones(Q.shape[1], 1, dtype=torch.float64, device=P.device) / Q.shape[1] if self.b is None else self.b
        fi = self.gamma / (self.gamma + self.epsilon)

        err = 1
        last_b = b.clone()
        iternum = 0
        while err > self.stoperr and iternum < self.numItermax:
            a = Pa / (Q @ b)
            b = Pb / (Q.t() @ a)
            if self.semi_use:
                b[:-1, :] = torch.pow(b[:-1, :], fi)
            err = torch.norm(b - last_b)
            last_b = b.clone()
            iternum += 1

        plan = n * a * Q * b.T
        self.b = b
       
        return plan[:, :-1].float()

def p2ot_assignment_and_quality(batch_cost, sinkhorn_model):
    B, N, K = batch_cost.shape
    
    assignments, qualities, Qs = [], [], []
    for b in range(B):
        Q = sinkhorn_model(batch_cost[b])
        Qs.append(Q)
        assignments.append(Q.argmax(dim=1))
        qualities.append(Q.sum(dim=1))
    return torch.stack(assignments), torch.stack(qualities), torch.stack(Qs)
class MaskedAutoencoderBNTF(nn.Module):
    def __init__(self, num_network=7,rho_0=0.8):
        super().__init__()
        self.num_network = num_network
        self.rho_0 = rho_0
        self.blocks = BNTF(depth=2, heads=4, dim_feedforward=2048)
        self.pos_embed = get_sinusoid_encoding_table(100,100)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 100))
        self.pred = nn.Sequential(nn.Linear(100, 1024), nn.LeakyReLU(), nn.Linear(1024, 100))
        self.sk_module = SemiCurrSinkhornKnopp(parcellation_type=num_network)
        self.initialize_weights()

    def initialize_weights(self):
        self.mask_token = torch.nn.init.xavier_normal_(self.mask_token)
        
    def get_subnetwork_mask(self, num_node=100, device='cpu'):
        subnetwork_map = {
            0: [1,2,3,4,5,6,7,8,9,51,52,53,54,55,56,57,58], #VN
            1: [10,11,12,13,14,15,59,60,61,62,63,64,65,66], #SMN
            2: [16,17,18,19,20,21,22,23,67,68,69,70,71,72,73], #DAN
            3: [24,25,26,27,28,29,30,74,75,76,77,78], #SAN
            4: [31,32,33,79,80], #LN
            5: [34,35,36,37,81,82,83,84,85,86,87,88,89], #CN
            6: [38,39,40,41,42,43,44,45,46,47,48,49,50,90,91,92,93,94,95,96,97,98,99,100], #DMN
        }
        mask = torch.zeros(self.num_network, num_node, dtype=torch.bool, device=device)
        for k, indices in subnetwork_map.items():
            mask[k, torch.tensor(indices) - 1] = 1
        return mask  # [K, 100]

    def compute_dynamic_rho(self, epoch, total_epoch=600):
        t = epoch
        T = total_epoch
        rho_0 = self.rho_0
        rho = rho_0 + (1 - rho_0) * math.exp(-5 * ((1 - t / T) ** 2))
        return rho, 1 - rho

    def get_features(self,input):
        with torch.no_grad():
            input = input.clone().detach()
            x = input
            x = self.blocks(x)

            return x

    @torch.no_grad()
    def get_cost_from_feature(self, x, model_ema):
        feature = model_ema.ema.get_features(x)  # [B, 100, D]
        B, N, D = feature.shape
        K = self.num_network
        subnet_mask = self.get_subnetwork_mask(num_node=N, device=x.device)  # [K, 100]
        subnet_mask = subnet_mask.unsqueeze(0).float().expand(B, -1, -1)     # [B, K, 100]

       
        denom = subnet_mask.sum(-1, keepdim=True) + 1e-6                    # [B, K, 1]
        # feature: [B, 100, D]
        subnet_feat = torch.bmm(subnet_mask, feature) / denom              # [B, K, D]

        # cosine similarity between each ROI and subnet prototype
        feature = F.normalize(feature, dim=-1)                              # [B, 100, D]
        subnet_feat = F.normalize(subnet_feat, dim=-1)                      # [B, K, D]
        cost = torch.matmul(feature, subnet_feat.transpose(1, 2))          # [B, 100, K]
        return cost

    def strategy_masking_eff(self, x,cost, mask_ratio, epoch):
        with torch.no_grad():
            rho, _ = self.compute_dynamic_rho(epoch)
            self.sk_module.rho = rho
            B, L, D = x.shape
           
            # === Step 2: Method "Confidence-Aware Soft Assignment" ===
            
            assign, quality, Q = p2ot_assignment_and_quality(cost, self.sk_module)
           

            # === Step 3: Method "Progressive Parcellation-Guided Masking" ===
            len_keep = int(L * (1 - mask_ratio))
            x_masked = torch.zeros((B, len_keep, D), device=x.device)
            mask = torch.ones(B, L, device=x.device)
            ids_restore = torch.zeros(B, L, dtype=torch.long, device=x.device)
            q = quality  # [B, L]
            low_mask = q < q.quantile(1 - rho, dim=1, keepdim=True)  # [B, L]
            num_low_tokens = low_mask.sum(dim=1)
            high_mask = ~low_mask
            noise = torch.zeros_like(q)
            rand_K = torch.rand(B, self.num_network, device=x.device)  # [B, K]
            weighted_noise = torch.bmm(Q, rand_K.unsqueeze(-1)).squeeze(-1)  # [B, L]
            noise[high_mask] = weighted_noise[high_mask].to(dtype=noise.dtype)
            ids_shuffle = noise.argsort(dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            ids_restore = ids_shuffle.argsort(dim=1)
            mask = torch.ones(B, L, device=x.device)
            mask.scatter_(1, ids_keep, 0)
            x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

            return x_masked, mask, ids_restore


    def forward_encoder(self, x, cost,mask_ratio, epoch):
        x, mask, ids_restore = self.strategy_masking_eff(x,cost, mask_ratio, epoch)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        x = self.blocks(x)
        x = self.pred(x)
        return x, mask

    def forward_loss(self, imgs, pred, mask):
        return ((pred - imgs) ** 2).mean(dim=-1).masked_select(mask.bool()).mean()

    def forward(self, imgs, mask_ratio=0.1, epoch = None, pretrain=True, deviated =False, model_ema = None):
        imgs = imgs.float()
        
        if pretrain:
            cost = self.get_cost_from_feature(imgs, model_ema)
            pred, mask = self.forward_encoder(imgs, cost,mask_ratio, epoch)
            loss = self.forward_loss(imgs, pred, mask)
            return loss
        else:
            x = imgs
            x = self.blocks(x)
            if not deviated:
                return x
            else:
                feature = x  # [B, 100, D]
                B, N, D = feature.shape
                K = self.num_network

                subnet_mask = self.get_subnetwork_mask(num_node=N, device=x.device)  # [K, N]
                subnet_mask = subnet_mask.unsqueeze(0).float().expand(B, -1, -1)     # [B, K, N]

                # subnet prototypes
                denom = subnet_mask.sum(-1, keepdim=True) + 1e-6
                subnet_feat = torch.bmm(subnet_mask, feature) / denom               # [B, K, D]

                # cosine similarity
                feature = F.normalize(feature, dim=-1)
                subnet_feat = F.normalize(subnet_feat, dim=-1)
                cost = torch.matmul(feature, subnet_feat.transpose(1, 2))           # [B, 100, K]
                _, _, Q = p2ot_assignment_and_quality(cost, self.sk_module)
                Q_template = self.get_subnetwork_mask(num_node=100, device=x.device).T.float()  # [100, K]
                delta_Q = Q - Q_template.unsqueeze(0) 
                template_label = Q_template.argmax(dim=1)  # [N]
                deviated = torch.zeros(B, K, K, device=x.device)
                for i in range(K):
                    idx = (template_label == i).nonzero(as_tuple=True)[0]
                    if len(idx) == 0:
                        continue
                    selected_Q = Q[:, idx, :]  # [B, num_i, K]
                    deviated[:, i, :] = selected_Q.sum(dim=1)

                

                return deviated
            
def mae_BNTF_base(num_network=7, rho_0=0.4):
    model = MaskedAutoencoderBNTF(num_network=num_network, rho_0=rho_0)
    print(model)
    return model

