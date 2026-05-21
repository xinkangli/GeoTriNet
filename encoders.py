"""
Neural network encoders for different modalities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NODE_FEAT_DIM = 30


def scatter_mean_pool(x, batch):
    """Scatter mean pooling."""
    ng = int(batch.max()) + 1
    out = torch.zeros(ng, x.size(1), dtype=x.dtype, device=x.device)
    cnt = torch.zeros(ng, 1, dtype=x.dtype, device=x.device)
    out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
    cnt.scatter_add_(0, batch.unsqueeze(1),
                     torch.ones(x.size(0), 1, dtype=x.dtype, device=x.device))
    return out / cnt.clamp(min=1)


class GATLayer(nn.Module):
    """Graph Attention layer."""
    
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.3):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.d = out_dim // n_heads
        
        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(1, n_heads, self.d * 2))
        nn.init.xavier_uniform_(self.attn.view(1, -1).unsqueeze(0))
        
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        n = x.size(0)
        H, dh = self.n_heads, self.d
        
        Ws = self.W_src(x).view(n, H, dh)
        Wd = self.W_dst(x).view(n, H, dh)
        e = torch.cat([Ws[src], Wd[dst]], -1)
        
        al = F.leaky_relu((e * self.attn).sum(-1), 0.2)
        al_exp = al.exp()
        den = torch.zeros(n, H, dtype=x.dtype, device=x.device)
        den.scatter_add_(0, dst.unsqueeze(1).expand(-1, H), al_exp)
        al_n = self.dropout(al_exp / (den[dst] + 1e-9))
        
        agg = torch.zeros(n, H, dh, dtype=x.dtype, device=x.device)
        agg.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, H, dh),
                         al_n.unsqueeze(-1) * Ws[src])
        
        return F.elu(self.ln(agg.reshape(n, H * dh)))


class AttentionPooling(nn.Module):
    """Attention-based pooling."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, batch):
        ng = int(batch.max()) + 1
        g = self.gate(x)
        num = torch.zeros(ng, x.size(1), dtype=x.dtype, device=x.device)
        den = torch.zeros(ng, 1, dtype=x.dtype, device=x.device)
        num.scatter_add_(0, batch.unsqueeze(1).expand_as(x), g * x)
        den.scatter_add_(0, batch.unsqueeze(1), g)
        return num / den.clamp(min=1e-9)


class GATEncoder2D(nn.Module):
    """2D Graph Attention Network encoder."""
    
    def __init__(self, node_dim=NODE_FEAT_DIM, hidden=256,
                 n_heads=4, n_layers=4, dropout=0.3):
        super().__init__()
        self.emb = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU()
        )
        self.layers = nn.ModuleList(
            [GATLayer(hidden, hidden, n_heads, dropout) for _ in range(n_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(n_layers)]
        )
        self.pool = AttentionPooling(hidden)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch_graph):
        x = self.emb(batch_graph["x"])
        for layer, ln in zip(self.layers, self.layer_norms):
            x = ln(x + layer(x, batch_graph["edge_index"]))
        return self.pool(self.dropout(x), batch_graph["batch"])


class DescEncoder(nn.Module):
    """Descriptor MLP encoder."""
    
    def __init__(self, fp_dim, h1=1024, h2=512, out=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, h1), nn.BatchNorm1d(h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h2, out), nn.BatchNorm1d(out), nn.GELU()
        )
    
    def forward(self, x):
        return self.net(x)


class EGNNLayer(nn.Module):
    """Equivariant Graph Neural Network layer."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        act = nn.SiLU()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim), act,
            nn.Linear(hidden_dim, hidden_dim), act
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), act,
            nn.Linear(hidden_dim, 1, bias=False),
            nn.Tanh()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), act,
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, h, pos, edge_index):
        src, dst = edge_index[0], edge_index[1]
        mask = src != dst
        src, dst = src[mask], dst[mask]
        
        diff = pos[dst] - pos[src]
        r2 = (diff.double() ** 2).sum(-1, keepdim=True).float().clamp(1e-6)
        
        raw = self.edge_mlp(torch.cat([h[src], h[dst], r2], -1))
        msg = raw * self.attn_mlp(raw)
        
        unit = diff / r2.sqrt()
        dp = torch.zeros_like(pos)
        if src.numel() > 0:
            dp.scatter_add_(0, dst.unsqueeze(-1).expand_as(diff),
                            unit * self.coord_mlp(msg))
        pos = pos + dp.clamp(-2., 2.)
        
        ag = torch.zeros_like(h)
        if src.numel() > 0:
            ag.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        
        h = self.ln(h + self.node_mlp(torch.cat([h, ag], -1)))
        
        if not h.isfinite().all():
            h = torch.nan_to_num(h, nan=0., posinf=1., neginf=-1.)
        
        return h, pos


class EGNNEncoder3D(nn.Module):
    """3D Equivariant Graph Neural Network encoder."""
    
    def __init__(self, node_dim=4, hidden=128, out=128,
                 n_layers=3, dropout=0.2):
        super().__init__()
        self.emb = nn.Linear(node_dim, hidden)
        self.emb_ln = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList([EGNNLayer(hidden) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden, out),
            nn.LayerNorm(out)
        )
    
    def forward(self, batch_graph):
        h = self.emb_ln(F.silu(self.emb(batch_graph["x"])))
        pos = batch_graph["pos"]
        edge_index = batch_graph["edge_index"]
        
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
        
        pooled = scatter_mean_pool(h, batch_graph["batch"])
        return self.proj(self.dropout(pooled))


class GatedFusion(nn.Module):
    """Gated multi-modality fusion."""
    
    def __init__(self, dims, fuse_out, dropout=0.3):
        super().__init__()
        total = sum(dims)
        mid = max(total // 2, fuse_out)
        
        self.gate = nn.Sequential(
            nn.Linear(total, mid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mid, len(dims)), nn.Softmax(dim=-1)
        )
        self.projs = nn.ModuleList([nn.Linear(d, fuse_out) for d in dims])
        self.norm = nn.LayerNorm(fuse_out)
        self.dropout = nn.Dropout(dropout)
        self.res_w = nn.Parameter(torch.ones(len(dims)) * 0.1)
    
    def forward(self, embeddings):
        projs = [self.projs[i](embeddings[i]) for i in range(len(embeddings))]
        w = self.gate(torch.cat(embeddings, -1))
        
        gated = sum(w[:, i:i+1] * projs[i] for i in range(len(embeddings)))
        rw = torch.sigmoid(self.res_w)
        res = sum(rw[i] * projs[i] for i in range(len(embeddings)))
        
        return self.dropout(self.norm(gated + res))


class TaskHead(nn.Module):
    """Task-specific prediction head."""
    
    def __init__(self, in_dim, hidden=256, out=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, out)
        )
    
    def forward(self, x):
        return self.net(x)
