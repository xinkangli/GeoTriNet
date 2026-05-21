"""
Molecular feature extraction utilities.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdDistGeom, rdForceFieldHelpers

RDLogger.DisableLog("rdApp.*")

# Constants
NODE_FEAT_DIM = 30
_SAFE_DESCS = [(n, f) for n, f in Descriptors.descList if n != "Ipc"]
_SYMBOL_MAP = {s: i for i, s in enumerate(['C','N','O','H','S','P','F'])}
_HYBRID_ORDER = ['SP','SP2','SP3','SP3D','SP3D2']
MAX_ATOMIC_NUM = 118
N_CONFS = 5


def smiles_to_morgan(smi, radius=2, bits=2048):
    """Extract Morgan fingerprint from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, bits),
                        dtype=np.float32)
    except:
        return None


def smiles_to_rdkit_desc(smi):
    """Extract RDKit descriptor features."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        vals = []
        for _, fn in _SAFE_DESCS:
            try:
                vals.append(float(fn(mol)))
            except:
                vals.append(np.nan)
        return np.nan_to_num(np.array(vals, np.float32), nan=0., posinf=0., neginf=0.)
    except:
        return None


def smiles_to_multiradius_fp(smi, radii=(1, 2, 3), n_bits=1024):
    """Multi-radius Morgan + RDKit descriptors."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(mol, r, n_bits),
                        dtype=np.float32) for r in radii]
        rd = smiles_to_rdkit_desc(smi)
        if rd is None:
            return None
        return np.concatenate(fps + [rd]).astype(np.float32)
    except:
        return None


def _encode_atom(atom):
    """Encode atomic features."""
    sym = atom.GetSymbol()
    sym_oh = [0] * 8
    sym_oh[_SYMBOL_MAP.get(sym, 7)] = 1
    
    val = min(atom.GetTotalValence(), 10)
    val_oh = [int(val == i) for i in range(11)]
    
    hyb = atom.GetHybridization().name
    hyb_oh = [int(hyb == h) for h in _HYBRID_ORDER]
    
    ar, ir, pr = atom.GetIsAromatic(), atom.IsInRing(), (sym == 'H')
    return (sym_oh + val_oh + hyb_oh +
            [int(ar), int(not ar), int(ir), int(not ir), int(pr), int(not pr)])


def smiles_to_graph_2d(smi, label=0):
    """Convert SMILES to 2D graph structure."""
    try:
        import torch
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        
        x = torch.tensor([_encode_atom(a) for a in mol.GetAtoms()], dtype=torch.float)
        s, d = [], []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            s += [i, j]
            d += [j, i]
        
        if not s:
            s, d = [0], [0]
        
        edge_index = torch.tensor([s, d], dtype=torch.long)
        return {"x": x, "edge_index": edge_index, "y": label}
    except:
        return None


def smiles_to_multi_conf_graphs(smi, label=0, n_confs=N_CONFS):
    """Generate multiple 3D conformers for a molecule."""
    try:
        import torch
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return []
        
        mol = Chem.AddHs(mol)
        prm = rdDistGeom.EmbedParameters()
        prm.randomSeed = 42
        prm.numThreads = 1
        
        ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=n_confs, params=prm)
        if len(ids) == 0:
            return []
        
        try:
            rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol)
        except:
            pass
        
        graphs = []
        for cid in ids:
            try:
                pos = mol.GetConformer(cid).GetPositions().astype(np.float32)
                atoms = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], np.int64)
                
                x = torch.zeros(len(atoms), 4, dtype=torch.float)
                for i, an in enumerate(atoms):
                    x[i, 0] = an / MAX_ATOMIC_NUM
                
                s, d = [], []
                for b in mol.GetBonds():
                    i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                    if i != j:
                        s += [i, j]
                        d += [j, i]
                
                if not s:
                    if len(atoms) >= 2:
                        s, d = [0, 1], [1, 0]
                    else:
                        s, d = [0], [0]
                
                graph = {
                    "x": x,
                    "edge_index": torch.tensor([s, d], dtype=torch.long),
                    "pos": torch.tensor(pos, dtype=torch.float),
                    "y": label
                }
                
                if _validate_graph(graph):
                    graphs.append(graph)
            except:
                continue
        
        return graphs
    except:
        return []


def _validate_graph(g):
    """Validate graph structure."""
    try:
        x = g["x"]
        ei = g["edge_index"]
        n = x.size(0)
        
        if n == 0:
            return False
        if not x.isfinite().all():
            return False
        if ei.dim() != 2 or ei.size(0) != 2:
            return False
        if ei.numel() > 0 and (ei.max() >= n or ei.min() < 0):
            return False
        if "pos" in g and g["pos"] is not None:
            if not g["pos"].isfinite().all():
                return False
        return True
    except:
        return False


def augment_smiles(smi, n=2):
    """Generate SMILES augmentations."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 2:
            return [smi]
        
        res, seen = [smi], {smi}
        roots = np.random.choice(mol.GetNumAtoms(),
                                 min(n*3, mol.GetNumAtoms()), replace=False)
        
        for r in roots:
            if len(res) >= n + 1:
                break
            try:
                aug = Chem.MolToSmiles(mol, rootedAtAtom=int(r), canonical=False)
                if aug and aug not in seen:
                    res.append(aug)
                    seen.add(aug)
            except:
                pass
        
        return res
    except:
        return [smi]
