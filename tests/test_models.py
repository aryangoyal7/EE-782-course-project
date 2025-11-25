"""
Tests for model components.
"""

import pytest
import torch
import numpy as np

from src.models.dinov3_wrapper import DINOv3Wrapper
from src.models.sam3d_wrapper import SAM3DWrapper
from src.models.fusion_mlp import FusionMLP


def test_dinov3_wrapper():
    """Test DINOv3 wrapper."""
    model = DINOv3Wrapper(model_name="dinov2_vitb14")
    model.eval()
    
    # Test single image
    img = torch.randn(1, 3, 224, 224)
    patch_tokens, pooled = model(img)
    
    assert patch_tokens.dim() == 2  # (B, N_patches, D) -> (B, N_patches*D) or similar
    assert pooled.dim() == 2  # (B, D)
    
    # Test multi-view
    img_multi = torch.randn(1, 4, 3, 224, 224)
    patch_tokens_multi, pooled_multi = model(img_multi)
    
    assert patch_tokens_multi.dim() == 3  # (B, N_views, N_patches, D)
    assert pooled_multi.dim() == 3  # (B, N_views, D)


def test_sam3d_wrapper():
    """Test SAM-3D wrapper."""
    model = SAM3DWrapper()
    model.eval()
    
    # Test forward pass
    img = torch.randn(1, 4, 3, 224, 224)
    geometry_features, mask_logits = model(img)
    
    assert geometry_features.shape[0] == 1  # Batch size
    assert geometry_features.shape[1] == model.feature_dim
    assert mask_logits.shape[0] == 1


def test_fusion_mlp():
    """Test Fusion MLP."""
    geometry_dim = 256
    semantic_dim = 768
    
    model = FusionMLP(
        geometry_dim=geometry_dim,
        semantic_dim=semantic_dim,
        hidden_dims=[128, 64],
        num_classes=10
    )
    
    # Test forward pass
    geometry_features = torch.randn(2, geometry_dim)
    semantic_features = torch.randn(2, semantic_dim)
    
    class_logits, volume_pred, voxel_logits = model(geometry_features, semantic_features)
    
    assert class_logits.shape == (2, 10)
    assert volume_pred.shape == (2, 1)
    assert voxel_logits.shape == (2, 64 * 64 * 64)


def test_fusion_mlp_heads():
    """Test individual heads of Fusion MLP."""
    model = FusionMLP(
        geometry_dim=256,
        semantic_dim=768,
        num_classes=10
    )
    
    geometry_features = torch.randn(1, 256)
    semantic_features = torch.randn(1, 768)
    
    # Test classification head
    class_logits = model.predict_class(geometry_features, semantic_features)
    assert class_logits.shape == (1, 10)
    
    # Test volume head
    volume_pred = model.predict_volume(geometry_features, semantic_features)
    assert volume_pred.shape == (1, 1)


def test_end_to_end_forward():
    """Test end-to-end forward pass through all models."""
    # Create models
    dinov3 = DINOv3Wrapper()
    sam3d = SAM3DWrapper()
    fusion = FusionMLP(
        geometry_dim=sam3d.feature_dim,
        semantic_dim=dinov3.feature_dim,
        num_classes=10
    )
    
    # Set to eval mode
    dinov3.eval()
    sam3d.eval()
    fusion.eval()
    
    # Create dummy input
    images = torch.randn(1, 4, 3, 224, 224)
    
    with torch.no_grad():
        # Extract features
        _, semantic_features = dinov3(images)
        semantic_features = semantic_features.mean(dim=1)  # Average over views
        
        geometry_features, _ = sam3d(images)
        
        # Fusion
        class_logits, volume_pred, voxel_logits = fusion(
            geometry_features, semantic_features
        )
    
    assert class_logits.shape[0] == 1
    assert volume_pred.shape[0] == 1
    assert voxel_logits.shape[0] == 1

