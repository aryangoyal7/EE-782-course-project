"""
Tests for inference pipeline.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

from src.models.dinov3_wrapper import DINOv3Wrapper
from src.models.sam3d_wrapper import SAM3DWrapper
from src.models.fusion_mlp import FusionMLP
from src.utils.metrics import compute_iou, compute_accuracy, compute_mape


def test_compute_iou():
    """Test IoU computation."""
    # Perfect match
    pred = np.ones((32, 32, 32))
    target = np.ones((32, 32, 32))
    iou = compute_iou(pred, target)
    assert abs(iou - 1.0) < 1e-6
    
    # No overlap
    pred = np.zeros((32, 32, 32))
    target = np.ones((32, 32, 32))
    iou = compute_iou(pred, target)
    assert abs(iou - 0.0) < 1e-6
    
    # Partial overlap
    pred = np.zeros((32, 32, 32))
    pred[:16, :, :] = 1.0
    target = np.zeros((32, 32, 32))
    target[:24, :, :] = 1.0
    iou = compute_iou(pred, target)
    assert 0.0 < iou < 1.0


def test_compute_accuracy():
    """Test accuracy computation."""
    pred = np.array([0, 1, 2, 0, 1])
    target = np.array([0, 1, 2, 1, 1])
    accuracy = compute_accuracy(pred, target)
    assert abs(accuracy - 0.8) < 1e-6


def test_compute_mape():
    """Test MAPE computation."""
    pred = np.array([100, 200, 300])
    target = np.array([110, 190, 310])
    mape = compute_mape(pred, target)
    assert mape > 0.0
    
    # Perfect prediction
    pred = np.array([100, 200, 300])
    target = np.array([100, 200, 300])
    mape = compute_mape(pred, target)
    assert abs(mape - 0.0) < 1e-6


def test_inference_pipeline():
    """Test basic inference pipeline."""
    # Create models
    dinov3 = DINOv3Wrapper()
    sam3d = SAM3DWrapper()
    fusion = FusionMLP(
        geometry_dim=sam3d.feature_dim,
        semantic_dim=dinov3.feature_dim,
        num_classes=10
    )
    
    dinov3.eval()
    sam3d.eval()
    fusion.eval()
    
    # Dummy input
    images = torch.randn(1, 4, 3, 224, 224)
    
    with torch.no_grad():
        # Extract features
        _, semantic_features = dinov3(images)
        semantic_features = semantic_features.mean(dim=1)
        geometry_features, _ = sam3d(images)
        
        # Predict
        class_logits, volume_pred, voxel_logits = fusion(
            geometry_features, semantic_features
        )
        
        pred_class = class_logits.argmax(dim=1)
        pred_volume = volume_pred.item()
        
        # Reshape voxels
        voxel_resolution = int(voxel_logits.shape[1] ** (1/3))
        pred_voxels = torch.sigmoid(voxel_logits).view(
            1, voxel_resolution, voxel_resolution, voxel_resolution
        )
    
    assert pred_class.shape == (1,)
    assert isinstance(pred_volume, float)
    assert pred_voxels.shape[1] == voxel_resolution

