"""
GeoTriNet: Geometry-Aware Trimodal Network
"""
import torch
import torch.nn as nn
from encoders import (
    GATEncoder2D, DescEncoder, EGNNEncoder3D,
    GatedFusion, TaskHead, scatter_mean_pool
)

NODE_FEAT_DIM = 30


class GeoTriNet(nn.Module):
    """
    Geometry-Aware Trimodal Network combining:
    - 2D graph attention (SMILES)
    - Descriptor features (Morgan + RDKit)
    - 3D geometric information (multi-conformer EGNN)
    """
    
    def __init__(self,
                 fp_dim,
                 gat_hidden=256,
                 desc_out=256,
                 egnn_hidden=128,
                 fuse_out=512,
                 head_hidden=256,
                 use_3d=True,
                 dropout=0.3):
        super().__init__()
        
        self.use_3d = use_3d
        self.fp_dim = fp_dim
        
        # Encoders
        self.gat = GATEncoder2D(NODE_FEAT_DIM, gat_hidden, 4, 4, dropout)
        self.desc = DescEncoder(fp_dim, 1024, 512, desc_out, dropout)
        
        dims = [gat_hidden, desc_out]
        
        if use_3d:
            self.egnn_single = EGNNEncoder3D(4, egnn_hidden, egnn_hidden, 3, dropout)
            self.conf_attn = nn.Sequential(
                nn.Linear(egnn_hidden, egnn_hidden // 2), nn.Tanh(),
                nn.Linear(egnn_hidden // 2, 1), nn.Sigmoid()
            )
            dims.append(egnn_hidden)
        
        # Fusion
        self.fusion = GatedFusion(dims, fuse_out, dropout)
        
        # Task heads (hierarchical: Contact & Oral → Overall)
        self.head_contact = TaskHead(fuse_out, head_hidden, 2, dropout)
        self.head_oral = TaskHead(fuse_out, head_hidden, 2, dropout)
        self.head_overall = TaskHead(fuse_out + 4, head_hidden, 2, dropout)
    
    def encode_3d_multiconf(self, conf_graphs_per_mol, device):
        """Encode multi-conformer graphs with attention pooling."""
        egnn_out = self.egnn_single.proj[0].out_features
        flat_graphs, conf_counts = [], []
        
        for conf_list in conf_graphs_per_mol:
            valid = [g for g in conf_list if g is not None]
            flat_graphs.extend(valid)
            conf_counts.append(len(valid))
        
        if not flat_graphs:
            return torch.zeros(len(conf_graphs_per_mol), egnn_out, device=device)
        
        try:
            bg = self._batch_graphs(flat_graphs, device)
            with torch.no_grad():
                all_e = self.egnn_single(bg)
        except:
            return torch.zeros(len(conf_graphs_per_mol), egnn_out, device=device)
        
        out_list, offset = [], 0
        for cnt in conf_counts:
            if cnt == 0:
                out_list.append(torch.zeros(egnn_out, device=device))
                continue
            
            mol_embs = all_e[offset:offset + cnt]
            offset += cnt
            
            if cnt == 1:
                out_list.append(mol_embs[0])
            else:
                w = self.conf_attn(mol_embs)
                out_list.append((w * mol_embs).sum(0) / w.sum(0).clamp(1e-9))
        
        return torch.stack(out_list, 0)
    
    def encode(self, bg2, fp_x, conf_graphs_per_mol=None):
        """Encode using all modalities."""
        embs = [self.gat(bg2), self.desc(fp_x)]
        
        if self.use_3d and conf_graphs_per_mol is not None:
            embs.append(self.encode_3d_multiconf(conf_graphs_per_mol, fp_x.device))
        
        return self.fusion(embs)
    
    def forward(self, bg2, fp_x, conf_graphs_per_mol=None):
        """Forward pass returning logits for all tasks."""
        z = self.encode(bg2, fp_x, conf_graphs_per_mol)
        
        lc = self.head_contact(z)
        lo = self.head_oral(z)
        
        # Hierarchical: Overall uses Contact & Oral representations
        co_sg = torch.cat([lc.detach(), lo.detach()], -1)
        lv = self.head_overall(torch.cat([z, co_sg], -1))
        
        return {"contact": lc, "oral": lo, "overall": lv}
    
    @staticmethod
    def _batch_graphs(graphs, device):
        """Batch multiple graphs."""
        xs, eis, bats = [], [], []
        has_pos = any(g.get("pos") is not None for g in graphs)
        pos_list = []
        off = 0
        
        for b, g in enumerate(graphs):
            n = g["x"].size(0)
            xs.append(g["x"])
            eis.append(g["edge_index"] + off)
            bats.append(torch.full((n,), b, dtype=torch.long))
            
            if has_pos:
                pos_list.append(g.get("pos") if g.get("pos") is not None
                               else torch.zeros(n, 3))
            off += n
        
        out = {
            "x": torch.cat(xs, 0).to(device),
            "edge_index": torch.cat(eis, 1).to(device),
            "batch": torch.cat(bats, 0).to(device)
        }
        
        if has_pos:
            out["pos"] = torch.cat(pos_list, 0).to(device)
        
        return out
