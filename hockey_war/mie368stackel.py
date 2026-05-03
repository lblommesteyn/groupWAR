import pandas as pd
import polars as pl
import numpy as np
import requests
import json
from tqdm import tqdm
import pickle
from itertools import combinations
from scipy.sparse import coo_matrix
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import RGCNConv
import seaborn as sns
import matplotlib.pyplot as plt
tqdm.pandas()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from zones import getzone
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.model_selection import KFold 
class DenseGCNLayer(nn.Module):
    '''
    No Longer Used
    '''
    def __init__(self, in_dim, out_dim, hidden_dim=None, eps_init=0.0,
                 train_eps=True, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, out_dim)

        self.norm = nn.LayerNorm(in_dim * 3)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self.eps = nn.Parameter(torch.tensor(eps_init)) if train_eps else torch.tensor(eps_init)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, A, X):
        B, N, F = X.shape

        A_pos = torch.clamp(A, min=0.0)
        A_neg = -torch.clamp(A, max=0.0)

        neigh_pos = torch.bmm(A_pos, X)
        neigh_neg = torch.bmm(A_neg, X)
        self_term = (1 + self.eps) * X

        concat = torch.cat([self_term, neigh_pos, neigh_neg], dim=-1)
        concat = self.norm(concat)

        flat = concat.reshape(B * N, -1)
        out = self.mlp(flat).view(B, N, -1)

        out = self.activation(out)
        out = self.dropout(out)
        return out
    
from torch_geometric.data import Batch

class GCN_DeepSet_AntiSym_Invariant(nn.Module):
    def __init__(self, node_in_dim, gcn_hidden, gcn_layers,
                 deepset_phi_dim, deepset_rho_dim,
                 n_nodes, vector_size, n_half=None, use_edge_agg=False,dropout = 0.1):
        #Intialize params
        super().__init__()
        self.N = n_nodes
        self.nh = n_half if n_half is not None else n_nodes // 2
        self.use_edge_agg = use_edge_agg
        
        self.embedder = nn.Linear(node_in_dim, vector_size)
        self.embedder2 = nn.Linear(vector_size, vector_size)

        layers = []
        for i in range(gcn_layers):
            in_d = vector_size if i == 0 else gcn_hidden
            layers.append(RGCNConv(in_d, gcn_hidden,num_relations=2))
            layers.append(nn.Linear(gcn_hidden, gcn_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(gcn_hidden, gcn_hidden))

        self.gcn = nn.ModuleList(layers)

        phi_in_dim = gcn_hidden + (1 if use_edge_agg else 0)
        self.phi = nn.Sequential(
            nn.Linear(phi_in_dim, deepset_phi_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(deepset_phi_dim, deepset_phi_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.rho = nn.Sequential(
            nn.Linear(deepset_phi_dim, deepset_rho_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(deepset_rho_dim, 1)
        )

        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(gcn_hidden)
        self.drop = nn.Dropout(dropout)
        self.sig = nn.Tanh()

    def forward(self, A, X, home_mask, away_mask=None):
        
        B,N, _ = X.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"
       # B = 1
        device = X.device
        home_mask = home_mask.to(device).float()
        if away_mask is None:
            away_mask = (1.0 - home_mask).to(device)
        else:
            away_mask = away_mask.to(device).float()
        H = self.relu(self.embedder(X))
        H = (self.embedder2(H)).reshape(B*N,-1)
        #A = A.reshape(B*N,-1)
        mask = A != 0
        b,i,j = (mask).nonzero(as_tuple=True)
        edge_index = torch.stack([b*N + i, b*N + j],dim=0).long()
        edge_weight = A[mask]
        edge_type = (A[mask]>0).long()
        agg = torch.zeros_like(H)
        for layer in self.gcn:
            if not (isinstance(layer,nn.Linear) or isinstance(layer,nn.Dropout) or isinstance(layer,nn.ReLU)):
                
                H = H+agg
                agg = torch.zeros_like(H)
                H2 = layer(H, edge_index, edge_type)
                
                row, col = edge_index
                msg = H2[col] * edge_weight.abs().unsqueeze(-1)

                agg.index_add_(0, row, msg)

                agg = self.relu(self.norm(agg))
                agg = self.drop(agg)#.reshape(B,N,-1)
            else:
                agg = (layer(agg))#.reshape(B*N,-1)
        H = (H+agg).reshape(B,N,-1)
        '''if self.use_edge_agg:
            neighbor_signal = (H.pow(2).sum(dim=2, keepdim=True))
            agg = torch.bmm(A, neighbor_signal).squeeze(-1).unsqueeze(-1)
            phi_input = torch.cat([H, agg], dim=2)
        else:'''
        phi_input = H

        BNN = B * N
        phi_in_flat = phi_input.view(BNN, -1)
        phi_out_flat = self.phi(phi_in_flat)
        phi_out = phi_out_flat.view(B, N, -1)

        home_mask_unsq = home_mask.unsqueeze(-1)
        away_mask_unsq = away_mask.unsqueeze(-1)

        home_sum = (phi_out * home_mask_unsq).sum(dim=1)
        away_sum = (phi_out * away_mask_unsq).sum(dim=1)

        home_scalar = self.rho(home_sum).squeeze(-1)
        away_scalar = self.rho(away_sum).squeeze(-1)

        out = home_scalar - away_scalar
        return 0.5 + 0.5*self.sig(out)
    
    def flipped_forward(self, A, X):
        A_flip = torch.flip(A, dims=(1,2))
        X_flip = torch.flip(X, dims=(1,))
        return self.forward(A_flip, X_flip)


class PeriodDataset(Dataset):
    def __init__(self, X, A, y, home_mask=None, away_mask=None):
       
        self.X = torch.tensor(X, dtype=torch.float32)
        self.A = torch.stack(A).to(dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        if home_mask is not None and away_mask is not None:
            self.home_mask = torch.tensor(home_mask, dtype=torch.bool)
            self.away_mask = torch.tensor(away_mask, dtype=torch.bool)
        else:
            self.home_mask = torch.tensor([1]*18 + [0]*18, dtype=torch.bool)
            self.away_mask = ~self.home_mask

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_sample = self.X[idx] 
        A_sample = self.A[idx] 
        y_sample = self.y[idx]
        return X_sample, A_sample, self.home_mask, self.away_mask, y_sample


def collate_fn(batch):
    Xs, As, home_masks, away_masks, ys = zip(*batch)

    Xs = torch.stack(Xs)    
    As = torch.stack(As)  
    home_masks = torch.stack(home_masks) 
    away_masks = torch.stack(away_masks)
    ys = torch.stack(ys).squeeze(-1)
    return Xs, As, home_masks, away_masks, ys


class ConstraintProjector:
    def __init__(self, n, n_half=None, tl_target=1, br_target=1,
                 neg_scale=1.25, max_iters=1, tol=1e-3):
        self.n = n
        self.n_half = n_half if n_half is not None else n // 2
        self.tl_target = tl_target
        self.br_target = br_target
        self.neg_scale = neg_scale
        self.max_iters = max_iters
        self.tol = tol

        nh = self.n_half

        self.tl_mask = torch.zeros(n, n, dtype=torch.bool)
        self.tl_mask[:nh, :nh] = 1
        self.tl_mask.fill_diagonal_(0)

        self.br_mask = torch.zeros(n, n, dtype=torch.bool)
        self.br_mask[nh:, nh:] = 1
        self.br_mask.fill_diagonal_(0)

        self.tr_mask = torch.zeros(n, n, dtype=torch.bool)
        self.tr_mask[:nh, nh:] = 1

        self.bl_mask = torch.zeros(n, n, dtype=torch.bool)
        self.bl_mask[nh:, :nh] = 1

        self.pos_mask = (self.tl_mask | self.br_mask).float()
        self.neg_mask = (self.tr_mask | self.bl_mask).float()

    def project(self, A):
        X = A.clone().float().to('cpu')
        nh = self.n_half
        if len(X.shape) == 3:
            for _ in range(self.max_iters):
                X_prev = X.clone()
                X.diagonal(dim1=1, dim2=2).zero_()#.fill_diagonal_(0)
                row_pos_sum = (X * self.pos_mask.unsqueeze(0)).sum(dim=1, keepdim=True)
                
                X[:,:nh,:nh] *= self.tl_target / ((X[:,:nh,:nh]).sum() + 1e-8)
                X[:,nh:,nh:] *= self.br_target / ((X[:,nh:,nh:]).sum() + 1e-8)

                row_pos_sum = (X * self.pos_mask.unsqueeze(0)).sum(dim=1, keepdim=True)
                row_neg_sum = (X * self.neg_mask.unsqueeze(0)).sum(dim=1, keepdim=True)
                scale_neg = - self.neg_scale*row_pos_sum / (row_neg_sum  + 1e-8)
                X = X * (self.neg_mask.unsqueeze(0) * scale_neg + (1 - self.neg_mask.unsqueeze(0)))

                # col_sums = X.sum(dim=0, keepdim=True) 

                # scale_col = -1*torch.maximum(-1*torch.ones_like(col_sums), self.tl_target / (-1*col_sums + 1e-8))
                row_pos_sum = (X * self.pos_mask.unsqueeze(0)).sum(dim=2, keepdim=True)
                row_neg_sum = (X * self.neg_mask.unsqueeze(0)).sum(dim=2, keepdim=True)
                scale_neg = - self.neg_scale*row_pos_sum / (row_neg_sum  + 1e-8)
                X = X * (self.neg_mask.unsqueeze(0) * scale_neg + (1 - self.neg_mask.unsqueeze(0)))
                # X *= scale_col

                X = 0.5 * (X + X.transpose(1,2))
                if torch.norm(X - X_prev) < self.tol:
                    break
        
            X[:,:18,:18] /= X[:,:18,:18].sum()*(1/(18**2))
            X[:,18:,18:] /= X[:,18:,18:].sum()*(1/(18**2))
            X[:,:18,18:] /= -X[:,:18,18:].sum()*(1/(18**2))
            X[:,18:,:18] /= -X[:,18:,:18].sum()*(1/(18**2))



            return X.to(device)
        else:
            for _ in range(self.max_iters):
                X_prev = X.clone()
                X.fill_diagonal_(0)

                row_pos_sum = (X * self.pos_mask).sum(dim=1, keepdim=True)
                
                X[:nh,:nh] *= self.tl_target / ((X[:nh,:nh]).sum() + 1e-8)
                X[nh:,nh:] *= self.br_target / ((X[nh:,nh:]).sum() + 1e-8)

                row_pos_sum = (X * self.pos_mask).sum(dim=1, keepdim=True)
                row_neg_sum = (X * self.neg_mask).sum(dim=1, keepdim=True)
                #print(-self.neg_scale*row_pos_sum / (row_neg_sum  + 1e-8))
                scale_neg = -self.neg_scale*row_pos_sum / (row_neg_sum  + 1e-8)
                X = X * (self.neg_mask * scale_neg + (1 - self.neg_mask))
                
                
                row_pos_sum = (X * self.pos_mask).sum(dim=0, keepdim=True)
                row_neg_sum = (X * self.neg_mask).sum(dim=0, keepdim=True)
                #print(-self.neg_scale*row_pos_sum / (row_neg_sum  + 1e-8))
                scale_neg = -self.neg_scale*row_pos_sum / (row_neg_sum  + 1e-8)
                X = X * (self.neg_mask * scale_neg + (1 - self.neg_mask))

                X = 0.5*(X + X.transpose(0,1))
                if torch.norm(X - X_prev) < self.tol:
                    break
        
            X[:18,:18] /= X[:18,:18].sum()*(1/(18**2))
            X[18:,18:] /= X[18:,18:].sum()*(1/(18**2))
            X[:18,18:] /= -X[:18,18:].sum()*(1/(18**2))
            X[18:,:18] /= -X[18:,:18].sum()*(1/(18**2))
            return X
        
projector = ConstraintProjector(36, n_half=18)

class StackelbergParamX(nn.Module):
    
    def __init__(self, n, mask_leader,mask_follower,S_init,n_half=None, tl_target=400*60, br_target=400*60,
                 neg_scale=1.25, max_iters=100, tol=1e-3):
        super().__init__()
        self.n = n
        self.n_half = n_half if n_half is not None else n // 2
        self.tl_target = tl_target
        self.br_target = br_target
        self.neg_scale = neg_scale
        self.max_iters = max_iters
        self.tol = tol

        
        self.mask_leader = mask_leader
        self.mask_follower = mask_follower
        self.S = nn.Parameter(S_init)
        self.tl_mask = torch.zeros(n, n, dtype=torch.bool)
        self.tl_mask[:n_half, :n_half] = 1
        self.tl_mask.fill_diagonal_(0)

        self.br_mask = torch.zeros(n, n, dtype=torch.bool)
        self.br_mask[n_half:, n_half:] = 1
        self.br_mask.fill_diagonal_(0)

        self.tr_mask = torch.zeros(n, n, dtype=torch.bool)
        self.tr_mask[:n_half, n_half:] = 1

        self.bl_mask = torch.zeros(n, n, dtype=torch.bool)
        self.bl_mask[n_half:, :n_half] = 1

        self.pos_mask = (self.tl_mask | self.br_mask).float()
        self.neg_mask = (self.tr_mask | self.bl_mask).float()
        self.relu = nn.ReLU()


    def build_X(self):
        X = self.S.clone()
        #print(X)
        nh = self.n_half
        X[:,:nh, :nh] = self.relu(X[:,:nh, :nh])
        X[:,nh:, nh:] = self.relu(X[:,nh:, nh:])
        X[:,:nh, nh:] = -self.relu(-X[:,:nh, nh:])
        X[:,nh:, :nh] = -self.relu(-X[:,nh:, :nh])
        return projector.project(X)

    def forward_leader(self,mask):
        X_full = self.S.clone()####
        X = X_full * (self.mask_leader-mask) + X_full.detach() * (1 - (self.mask_leader-mask))
        return X

    def forward_follower(self):
        X_full = self.build_X()#.S.clone()#.build_X()###
        X = X_full * self.mask_follower + X_full.detach() * (1 - self.mask_follower)
        return X
def stackelberg(Y_fixed,model,mask,avg = None,opt_leader = None,opt_follower = None):
    '''
    Adversarial optimization loop for adjacency matrix
    '''
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(42)
    n = 36
    n_2 = n//2
    mask_leader = torch.zeros(n, n).to(device).int().unsqueeze(0)
    mask_leader[:,:n_2, :n_2] = True
    mask_follower = 1 - mask_leader
    nh = n_2
    #Initialize if not using starting point
    B = Y_fixed.shape[0]
    if avg == None:
        S_init = torch.ones(B,n, n)
        S_init[:]/=S_init.sum()
        S_init[:,nh:,:nh] *= -1.25
        S_init[:,:nh,nh:] *= -1.25
    else:
        S_init = avg.clone()
    steps = (100 if avg == None else 50)
    S_init[mask==1] = 0
    #Start params
    stack = StackelbergParamX(n,mask_leader,mask_follower,S_init).to(device)
    wd = 1e-4
    coef = 0
    #Define optmizers
    if avg == None:
        opt_leader = torch.optim.Adamax([stack.S], lr=1e-5,weight_decay = wd,betas=(0.9,0.99),eps=1e-5)#torch.optim.SGD([stack.S], lr=1e-5,momentum=0.9,nesterov=True)#,weight_decay = wd)#,betas=(0.8,0.97))#,momentum=0.9)#,nesterov=True)#,weight_decay=0)  
        opt_follower = torch.optim.Adamax([stack.S], lr=5e-4,weight_decay = wd,betas=(0.8,0.98),eps=1e-5)#torch.optim.SGD([stack.S], lr=5e-5,momentum=0.9,nesterov=True)#torch.optim.Adam([stack.S], lr=5e-4,weight_decay = wd)#,betas=(0.8,0.97))# sched_l = torch.optim.lr_scheduler.CosineAnnealingLR(opt_leader, T_max=steps)
    else: 
        opt_leader = torch.optim.Adamax([stack.S], lr=1e-5,weight_decay = wd,betas=(0.9,0.99),eps=1e-5)#torch.optim.SGD([stack.S], lr=1e-5,momentum=0.9,nesterov=True)#,weight_decay = wd)#,betas=(0.8,0.97))#,momentum=0.9)#,nesterov=True)#,weight_decay=0)  
        opt_follower = torch.optim.Adamax([stack.S], lr=5e-4,weight_decay = wd,betas=(0.8,0.98),eps=1e-5)#torch.optim.SGD([stack.S], lr=5e-5,momentum=0.9,nesterov=True)#torch.optim.Adam([stack.S], lr=5e-4,weight_decay = wd)#,betas=(0.8,0.97))# sched_l = torch.optim.lr_scheduler.CosineAnnealingLR(opt_leader, T_max=steps)
    sched_l = torch.optim.lr_scheduler.CosineAnnealingLR(opt_leader, T_max=steps)
    sched_f = torch.optim.lr_scheduler.CosineAnnealingLR(opt_follower, T_max=steps//5)
    home_mask = torch.tensor([1]*18 + [0]*18, dtype=torch.bool).unsqueeze(0)#.to(device)
    Y_fixed = Y_fixed.to(device)#.unsqueeze(0)   
    mask = mask.to(device)
    #model.eval()    
    #Run loop with early stopping
    prev_loss = None
    patience = 0
    for step in range(steps):
        #optmize for home team
        X_leader = stack.forward_leader(mask)
        loss = torch.stack([-m(X_leader, Y_fixed, home_mask).mean() for m in model]).mean()
        opt_leader.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([stack.S], max_norm=1)
        opt_leader.step()

        # Early stopping on leader loss
        loss_val = loss.item()
        if prev_loss is not None and abs(loss_val - prev_loss) < 1e-5:
            patience += 1
            if patience >= 5:
                break
        else:
            patience = 0
        prev_loss = loss_val

        #Optmize follower team less often
        if (step+1)%5 == 0:
            X_follower = stack.forward_follower()
            loss = torch.stack([m(X_follower, Y_fixed, home_mask).mean() for m in model]).mean()
            opt_follower.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([stack.S], max_norm=1)
            opt_follower.step()
            sched_f.step()

        sched_l.step()
    #Return (no gradients needed for final eval)
    X_final = stack.build_X().detach()
    with torch.no_grad():
        final_preds = torch.stack([m(X_final, Y_fixed, home_mask) for m in model])
    return final_preds.mean(dim=0).to('cpu'), X_final.to('cpu'), opt_leader, opt_follower
