"""
GeoTriNet - Geometry-Aware Trimodal Network for Toxicity Prediction
"""

__version__ = "0.1.0"
__author__ = "GeoTriNet Contributors"

from .models import GeoTriNet
from .encoders import GATEncoder2D, EGNNEncoder3D, DescEncoder
from .train import train_hierarchical
from .features import smiles_to_graph_2d, smiles_to_multi_conf_graphs
from .data import load_joint_dataset, build_joint_features
from .metrics import compute_metrics

__all__ = [
    "GeoTriNet",
    "GATEncoder2D",
    "EGNNEncoder3D", 
    "DescEncoder",
    "train_hierarchical",
    "smiles_to_graph_2d",
    "smiles_to_multi_conf_graphs",
    "load_joint_dataset",
    "build_joint_features",
    "compute_metrics",
]
