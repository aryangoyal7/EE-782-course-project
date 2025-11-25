"""
Evaluation metrics for 3D object understanding.
"""

import numpy as np
import torch
from typing import Dict


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) for binary voxel grids.
    
    Args:
        pred: Predicted binary voxel grid (H, W, D)
        target: Ground truth binary voxel grid (H, W, D)
    
    Returns:
        IoU score in [0, 1]
    """
    pred_binary = (pred > 0.5).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)
    
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(np.maximum(pred_binary, target_binary))
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        pred: Predicted class indices (N,)
        target: Ground truth class indices (N,)
    
    Returns:
        Accuracy score in [0, 1]
    """
    return np.mean(pred == target)


def compute_mape(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE) for volume regression.
    
    Args:
        pred: Predicted volumes (N,)
        target: Ground truth volumes (N,)
    
    Returns:
        MAPE score (percentage)
    """
    # Avoid division by zero
    mask = target != 0
    if not np.any(mask):
        return 0.0
    
    mape = np.mean(np.abs((target[mask] - pred[mask]) / target[mask])) * 100
    return mape


def compute_volume(voxels: np.ndarray, voxel_size: float = 1.0) -> float:
    """
    Compute volume from voxel grid.
    
    Args:
        voxels: Binary voxel grid (H, W, D)
        voxel_size: Size of each voxel in world units
    
    Returns:
        Volume in cubic units
    """
    occupied_voxels = np.sum(voxels > 0.5)
    return occupied_voxels * (voxel_size ** 3)


def evaluate_batch(
    pred_voxels: torch.Tensor,
    target_voxels: torch.Tensor,
    pred_classes: torch.Tensor,
    target_classes: torch.Tensor,
    pred_volumes: torch.Tensor,
    target_volumes: torch.Tensor
) -> Dict[str, float]:
    """
    Compute all metrics for a batch.
    
    Args:
        pred_voxels: Predicted voxel grids (B, H, W, D)
        target_voxels: Ground truth voxel grids (B, H, W, D)
        pred_classes: Predicted class indices (B,)
        target_classes: Ground truth class indices (B,)
        pred_volumes: Predicted volumes (B,)
        target_volumes: Ground truth volumes (B,)
    
    Returns:
        Dictionary of metric names and values
    """
    # Convert to numpy
    pred_voxels_np = pred_voxels.detach().cpu().numpy()
    target_voxels_np = target_voxels.detach().cpu().numpy()
    pred_classes_np = pred_classes.detach().cpu().numpy()
    target_classes_np = target_classes.detach().cpu().numpy()
    pred_volumes_np = pred_volumes.detach().cpu().numpy()
    target_volumes_np = target_volumes.detach().cpu().numpy()
    
    # Compute IoU for each sample
    ious = []
    for i in range(pred_voxels_np.shape[0]):
        iou = compute_iou(pred_voxels_np[i], target_voxels_np[i])
        ious.append(iou)
    
    mean_iou = np.mean(ious)
    accuracy = compute_accuracy(pred_classes_np, target_classes_np)
    mape = compute_mape(pred_volumes_np, target_volumes_np)
    
    return {
        'iou': mean_iou,
        'accuracy': accuracy,
        'mape': mape
    }

