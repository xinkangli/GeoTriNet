"""
Evaluation metrics.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score
)


def compute_metrics(y_true, y_pred, y_proba):
    """
    Compute classification metrics.
    
    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        y_proba: predicted probabilities (positive class)
    
    Returns:
        dict with ACC, AUC, Specificity, Sensitivity, F1
    """
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except:
        return dict(ACC=np.nan, AUC=np.nan, SP=np.nan, SE=np.nan, F1=np.nan)
    
    try:
        auc = roc_auc_score(y_true, y_proba)
    except:
        auc = np.nan
    
    return dict(
        ACC=round(accuracy_score(y_true, y_pred), 4),
        AUC=round(auc, 4),
        SP=round(tn / (tn + fp + 1e-9), 4),
        SE=round(tp / (tp + fn + 1e-9), 4),
        F1=round(f1_score(y_true, y_pred, zero_division=0), 4)
    )
