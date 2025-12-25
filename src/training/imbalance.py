# src/training/imbalance.py
import numpy as np
import torch
from torch.utils.data import Subset

def make_imbalanced_subset(dataset, target_ratio=0.1, seed=42):
    """Reduziert positive Samples auf target_ratio (z.B. 0.1)."""
    np.random.seed(seed)
    
    # PCam-Spezifischer Check für Speed:
    # torchvision.datasets.PCAM hat oft ein Attribut _labels oder man muss 
    # auf das unterliegende HDF5 zugreifen. 
    # Falls das nicht klappt, nutzen wir deine Liste:
    if hasattr(dataset, '_labels'):
        targets = np.array(dataset._labels)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    else:
        print("⌛ Extrahiere Labels (das kann einen Moment dauern)...")
        # Wir laden nur die Labels, nicht die Bilder
        targets = np.array([y for _, y in dataset]) 

    indices = np.arange(len(targets))
    pos_indices = indices[targets == 1].flatten()
    neg_indices = indices[targets == 0].flatten()
    
    n_neg = len(neg_indices)
    n_pos_needed = int((target_ratio * n_neg) / (1 - target_ratio))
    
    if n_pos_needed > len(pos_indices):
        return dataset, 1.0

    keep_pos = np.random.choice(pos_indices, n_pos_needed, replace=False)
    new_indices = np.concatenate([neg_indices, keep_pos])
    np.random.shuffle(new_indices)
    
    subset = Subset(dataset, new_indices)
    pos_weight = n_neg / n_pos_needed # Verhältnis Neg/Pos
    
    return subset, pos_weight