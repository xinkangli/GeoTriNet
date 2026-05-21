"""
Training engine for GeoTriNet.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from models import GeoTriNet
from metrics import compute_metrics
from features import augment_smiles, smiles_to_graph_2d, smiles_to_multi_conf_graphs


TASK_NAMES = ["Contact", "Oral", "Overall"]


def focal_loss(logits, targets, class_weights, gamma=2.0, label_smooth=0.1):
    """Focal loss with label smoothing."""
    n = logits.size(-1)
    lp = F.log_softmax(logits, -1)
    pt = lp.exp()[torch.arange(len(targets)), targets].detach()
    fw = (1 - pt) ** gamma
    
    if label_smooth > 0:
        oh = F.one_hot(targets, n).float()
        q = oh * (1 - label_smooth) + label_smooth / n
        nll = -(q * lp).sum(-1)
    else:
        nll = F.nll_loss(lp, targets, reduction="none")
    
    return (fw * nll * class_weights[targets]).mean()


def rdrop_focal(logits1, logits2, targets, class_weights, gamma=2.0, 
                label_smooth=0.1, alpha=0.5):
    """R-Drop with focal loss."""
    tl = (focal_loss(logits1, targets, class_weights, gamma, label_smooth) +
          focal_loss(logits2, targets, class_weights, gamma, label_smooth)) * 0.5
    
    p1 = F.softmax(logits1, -1)
    p2 = F.softmax(logits2, -1)
    kl = (F.kl_div(p1.log(), p2.detach(), reduction="batchmean") +
          F.kl_div(p2.log(), p1.detach(), reduction="batchmean")) * 0.5
    
    return tl + alpha * kl


def _get_class_weights(labels, device):
    """Compute class weights from label imbalance."""
    pw = float((labels == 0).sum()) / max(float((labels == 1).sum()), 1.)
    return torch.tensor([1., pw], dtype=torch.float32, device=device)


def _lr_schedule(epoch, warmup=5, total=120):
    """Warmup + cosine annealing learning rate schedule."""
    if epoch < warmup:
        return epoch / max(warmup, 1)
    p = (epoch - warmup) / max(total - warmup, 1)
    return max(0.05, 0.5 * (1 + math.cos(math.pi * p)))


def _batch_graphs(graphs, device):
    """Batch multiple graph structures."""
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


def train_hierarchical(
    model_fn,
    smiles,
    labels_mat,
    fp_X,
    conf_graphs=None,
    n_epochs=120,
    batch_size=16,
    lr=3e-4,
    device="cpu",
    aug_n=2,
    label_smooth=0.1,
    focal_gamma=2.0,
    rdrop_alpha=0.5,
    warmup_ep=5,
    train_seeds=(42,),
    task_weights=None
):
    """
    Train GeoTriNet model.
    
    Args:
        model_fn: function returning a GeoTriNet instance
        smiles: list of SMILES strings
        labels_mat: array of shape (N, 3) with labels for [Contact, Oral, Overall]
        fp_X: descriptor features array of shape (N, fp_dim)
        conf_graphs: list of conformer graphs (optional)
        ... (other hyperparameters)
    
    Returns:
        list of result dictionaries
    """
    N = len(smiles)
    assert labels_mat.shape == (N, 3), f"labels_mat shape {labels_mat.shape} != (N, 3)"
    
    tw = (torch.ones(3, device=device) if task_weights is None
          else torch.tensor(task_weights, dtype=torch.float32, device=device))
    
    # Class weights for each task
    cews = []
    for t in range(3):
        valid = ~np.isnan(labels_mat[:, t])
        cews.append(_get_class_weights(labels_mat[valid, t].astype(int), device))
    
    # 8:1:1 split (train:val:test)
    valid_c = np.where(~np.isnan(labels_mat[:, 0]))[0]
    tr_c, tmp_c = train_test_split(
        valid_c, test_size=0.2, random_state=42,
        stratify=labels_mat[valid_c, 0].astype(int)
    )
    va_c, te_c = train_test_split(
        tmp_c, test_size=0.5, random_state=42,
        stratify=labels_mat[tmp_c, 0].astype(int)
    )
    
    va_set = set(va_c.tolist())
    te_set = set(te_c.tolist())
    tr_set = set(range(N)) - va_set - te_set
    
    vai_arr = np.array(sorted(va_set))
    tei_arr = np.array(sorted(te_set))
    tri_arr = np.array(sorted(tr_set))
    
    print(f"Data split: train={len(tri_arr)}, val={len(vai_arr)}, test={len(tei_arr)}")
    
    # Build 2D graphs
    print("Building 2D graphs...")
    g2_all = []
    for smi in smiles:
        g = smiles_to_graph_2d(smi)
        g2_all.append(g if g is not None else None)
    
    # SMILES augmentation for training set
    print(f"Augmenting SMILES (n={aug_n})...")
    aug_g2, aug_fp, aug_lbl = [], [], []
    aug_conf = []
    
    for pos in tri_arr:
        if g2_all[pos] is None:
            continue
        
        cands = augment_smiles(smiles[pos], aug_n)
        added = 0
        
        for c in cands:
            g = smiles_to_graph_2d(c)
            if g is not None:
                aug_g2.append(g)
                aug_fp.append(fp_X[pos])
                aug_lbl.append(labels_mat[pos])
                if conf_graphs is not None:
                    aug_conf.append(conf_graphs[pos])
                added += 1
        
        if added == 0:
            aug_g2.append(g2_all[pos])
            aug_fp.append(fp_X[pos])
            aug_lbl.append(labels_mat[pos])
            if conf_graphs is not None:
                aug_conf.append(conf_graphs[pos])
    
    aug_fp = np.array(aug_fp, dtype=np.float32)
    aug_lbl = np.array(aug_lbl, dtype=np.float32)
    
    # Prepare val & test sets
    val_g2 = [g2_all[i] for i in vai_arr if g2_all[i] is not None]
    val_mask = [i for i in vai_arr if g2_all[i] is not None]
    val_fp = fp_X[val_mask]
    val_lbl = labels_mat[val_mask]
    val_conf = [conf_graphs[i] for i in val_mask] if conf_graphs is not None else None
    
    test_g2 = [g2_all[i] for i in tei_arr if g2_all[i] is not None]
    test_mask = [i for i in tei_arr if g2_all[i] is not None]
    test_fp = fp_X[test_mask]
    test_lbl = labels_mat[test_mask]
    test_conf = [conf_graphs[i] for i in test_mask] if conf_graphs is not None else None
    
    print(f"Augmented train={len(aug_g2)}, val={len(val_g2)}, test={len(test_g2)}")
    
    # Training loop
    per_task_test_probs = [[] for _ in range(3)]
    test_true_final = [[] for _ in range(3)]
    
    for seed in train_seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = model_fn().to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: _lr_schedule(ep, warmup_ep, n_epochs)
        )
        
        best_val_loss = float("inf")
        patience = 0
        PATIENCE = 20
        
        for epoch in range(1, n_epochs + 1):
            # Training epoch
            model.train()
            indices = np.random.permutation(len(aug_g2))
            
            for s in range(0, len(aug_g2), batch_size):
                bi = indices[s:s + batch_size]
                if len(bi) < 2:
                    continue
                
                bg2 = _batch_graphs([aug_g2[i] for i in bi], device)
                bfp = torch.tensor(aug_fp[bi], dtype=torch.float32, device=device)
                bconf = [aug_conf[i] for i in bi] if conf_graphs is not None else None
                blbl = aug_lbl[bi]
                
                with torch.enable_grad():
                    output = model(bg2, bfp, bconf)
                    
                    batch_loss = torch.tensor(0., device=device)
                    for t, tkey in enumerate(["contact", "oral", "overall"]):
                        valid = ~np.isnan(blbl[:, t])
                        if valid.sum() == 0:
                            continue
                        
                        vi = np.where(valid)[0]
                        by = torch.tensor(blbl[vi, t].astype(int),
                                        dtype=torch.long, device=device)
                        lo_t = output[tkey][vi]
                        
                        tl = focal_loss(lo_t, by, cews[t], focal_gamma, label_smooth)
                        batch_loss = batch_loss + tw[t] * tl
                
                if batch_loss.isfinite():
                    optimizer.zero_grad()
                    batch_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
            
            # Validation epoch
            model.eval()
            val_loss = 0.
            val_probs = [[] for _ in range(3)]
            val_true = [[] for _ in range(3)]
            
            with torch.no_grad():
                for s in range(0, len(val_g2), batch_size):
                    bi = list(range(s, min(s + batch_size, len(val_g2))))
                    
                    bg2 = _batch_graphs([val_g2[i] for i in bi], device)
                    bfp = torch.tensor(val_fp[bi], dtype=torch.float32, device=device)
                    bconf = [val_conf[i] for i in bi] if val_conf is not None else None
                    blbl = val_lbl[bi]
                    
                    output = model(bg2, bfp, bconf)
                    
                    for t, tkey in enumerate(["contact", "oral", "overall"]):
                        valid = ~np.isnan(blbl[:, t])
                        if valid.sum() == 0:
                            continue
                        
                        vi = np.where(valid)[0]
                        by = torch.tensor(blbl[vi, t].astype(int),
                                        dtype=torch.long, device=device)
                        lo_t = output[tkey][vi]
                        
                        tl = focal_loss(lo_t, by, cews[t], focal_gamma, label_smooth)
                        val_loss += tl.item() * len(vi)
                        
                        probs = F.softmax(lo_t, -1)[:, 1].cpu().numpy()
                        val_probs[t].extend(probs)
                        val_true[t].extend(blbl[vi, t].astype(int).tolist())
            
            scheduler.step()
            
            val_loss /= max(len(val_g2), 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Test inference
        model.eval()
        test_probs = [[] for _ in range(3)]
        test_true = [[] for _ in range(3)]
        
        with torch.no_grad():
            for s in range(0, len(test_g2), batch_size):
                bi = list(range(s, min(s + batch_size, len(test_g2))))
                
                bg2 = _batch_graphs([test_g2[i] for i in bi], device)
                bfp = torch.tensor(test_fp[bi], dtype=torch.float32, device=device)
                bconf = [test_conf[i] for i in bi] if test_conf is not None else None
                blbl = test_lbl[bi]
                
                output = model(bg2, bfp, bconf)
                
                for t, tkey in enumerate(["contact", "oral", "overall"]):
                    valid = ~np.isnan(blbl[:, t])
                    if valid.sum() == 0:
                        continue
                    
                    vi = np.where(valid)[0]
                    lo_t = output[tkey][vi]
                    
                    probs = F.softmax(lo_t, -1)[:, 1].cpu().numpy()
                    test_probs[t].extend(probs)
                    test_true[t].extend(blbl[vi, t].astype(int).tolist())
        
        for t in range(3):
            if test_probs[t]:
                per_task_test_probs[t].append(np.array(test_probs[t]))
                test_true_final[t] = np.array(test_true[t])
    
    # Ensemble results
    rows = []
    for t, tn in enumerate(TASK_NAMES):
        if not per_task_test_probs[t]:
            continue
        
        avg_p = np.mean(per_task_test_probs[t], axis=0)
        metrics = compute_metrics(test_true_final[t], 
                                 (avg_p >= 0.5).astype(int), 
                                 avg_p)
        
        print(f"{tn}: ACC={metrics['ACC']:.4f}, AUC={metrics['AUC']:.4f}")
        rows.append({"task": tn, **metrics})
    
    return rows
