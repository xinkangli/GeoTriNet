"""
Data loading and preprocessing utilities.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from features import smiles_to_multiradius_fp


def load_dataset(path, exclude_source=None):
    """
    Load dataset from CSV.
    
    Args:
        path: path to CSV file
        exclude_source: exclude rows where 'source' column contains this string
    
    Returns:
        (smiles_list, labels_array)
    """
    df = pd.read_csv(path)
    
    if exclude_source and "source" in df.columns:
        df = df[~df["source"].str.contains(exclude_source, na=False)]
    
    return df["SMILES"].tolist(), df["label"].values.astype(int)


def load_joint_dataset(contact_path, oral_path, overall_path, exclude_source=None):
    """
    Load and merge datasets from multiple files.
    
    Returns:
        DataFrame with columns: SMILES, label_Contact, label_Oral, label_Overall
    """
    dsets = {}
    for name, path in [("Contact", contact_path), 
                       ("Oral", oral_path), 
                       ("Overall", overall_path)]:
        df = pd.read_csv(path)
        
        if exclude_source and "source" in df.columns:
            df = df[~df["source"].str.contains(exclude_source, na=False)]
        
        df = df[["SMILES", "label"]]
        dsets[name] = df.rename(columns={"label": f"label_{name}"})
    
    df_joint = dsets["Contact"]
    for name in ["Oral", "Overall"]:
        df_joint = pd.merge(df_joint, dsets[name], on="SMILES", how="outer")
    
    return df_joint


def build_joint_features(smiles_list, labels_mat):
    """
    Build descriptor features and label matrix.
    
    Args:
        smiles_list: list of SMILES strings
        labels_mat: array of shape (N, 3) with labels for Contact, Oral, Overall
    
    Returns:
        (fp_scaled, labels_ok, smiles_ok, task_weights)
    """
    print(f"Building multi-radius fingerprints for {len(smiles_list)} molecules...")
    
    fp_list = []
    ok_idx = []
    
    for i, smi in enumerate(smiles_list):
        if i % 200 == 0:
            print(f"  {i}/{len(smiles_list)}")
        
        fp = smiles_to_multiradius_fp(smi)
        if fp is not None and np.all(np.isfinite(fp)):
            fp_list.append(fp)
            ok_idx.append(i)
        else:
            fp_list.append(None)
    
    # Keep only valid indices
    ok_idx = [i for i, x in enumerate(fp_list) if x is not None]
    fp_raw = np.array([fp_list[i] for i in ok_idx], dtype=np.float32)
    labels_ok = labels_mat[ok_idx]
    smiles_ok = [smiles_list[i] for i in ok_idx]
    
    # Standardize features
    scaler = StandardScaler()
    fp_scaled = scaler.fit_transform(fp_raw).astype(np.float32)
    
    # Compute task weights
    n_valid = np.array([np.nansum(~np.isnan(labels_ok[:, t])) 
                       for t in range(3)], float)
    task_weights = 1.0 / np.sqrt(n_valid)
    task_weights = (task_weights / task_weights.sum() * 3).astype(np.float32)
    
    print(f"Valid molecules: {len(ok_idx)}")
    print(f"Feature dimension: {fp_scaled.shape[1]}")
    
    return fp_scaled, labels_ok, smiles_ok, task_weights
