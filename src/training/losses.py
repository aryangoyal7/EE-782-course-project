"""
Loss functions for training the fusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassificationLoss(nn.Module):
    """Cross-entropy loss for classification."""
    
    def __init__(self, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            pred_logits: Predicted class logits (B, num_classes)
            target: Ground truth class indices (B,)
        
        Returns:
            Loss value
        """
        return self.criterion(pred_logits, target)


class VolumeLoss(nn.Module):
    """Smooth L1 loss for volume regression."""
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.criterion = nn.SmoothL1Loss(beta=beta)
    
    def forward(self, pred_volume: torch.Tensor, target_volume: torch.Tensor) -> torch.Tensor:
        """
        Compute volume regression loss.
        
        Args:
            pred_volume: Predicted volumes (B, 1)
            target_volume: Ground truth volumes (B,)
        
        Returns:
            Loss value
        """
        target_volume = target_volume.unsqueeze(1)  # (B,) -> (B, 1)
        return self.criterion(pred_volume, target_volume)


class ConsistencyLoss(nn.Module):
    """
    Consistency loss: penalizes mismatch between projected 3D mask and 2D mask from DINO.
    """
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        projected_3d_mask: torch.Tensor,
        dino_2d_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss.
        
        Args:
            projected_3d_mask: Projected 3D mask logits (B, H, W)
            dino_2d_mask: 2D mask from DINO features (B, H, W)
        
        Returns:
            Loss value
        """
        return self.criterion(projected_3d_mask, dino_2d_mask)


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning.
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        volume_weight: float = 1.0,
        consistency_weight: float = 0.1,
        use_consistency: bool = True
    ):
        """
        Initialize combined loss.
        
        Args:
            classification_weight: Weight for classification loss
            volume_weight: Weight for volume regression loss
            consistency_weight: Weight for consistency loss
            use_consistency: Whether to use consistency loss
        """
        super().__init__()
        
        self.classification_loss = ClassificationLoss()
        self.volume_loss = VolumeLoss()
        self.consistency_loss = ConsistencyLoss()
        
        self.classification_weight = classification_weight
        self.volume_weight = volume_weight
        self.consistency_weight = consistency_weight
        self.use_consistency = use_consistency
    
    def forward(
        self,
        pred_class_logits: torch.Tensor,
        target_class: torch.Tensor,
        pred_volume: torch.Tensor,
        target_volume: torch.Tensor,
        projected_3d_mask: Optional[torch.Tensor] = None,
        dino_2d_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            pred_class_logits: Predicted class logits (B, num_classes)
            target_class: Ground truth class indices (B,)
            pred_volume: Predicted volumes (B, 1)
            target_volume: Ground truth volumes (B,)
            projected_3d_mask: Optional projected 3D mask logits (B, H, W)
            dino_2d_mask: Optional 2D mask from DINO (B, H, W)
        
        Returns:
            Dictionary with individual losses and total loss
        """
        # Classification loss
        cls_loss = self.classification_loss(pred_class_logits, target_class)
        
        # Volume loss
        vol_loss = self.volume_loss(pred_volume, target_volume)
        
        # Consistency loss (optional)
        cons_loss = torch.tensor(0.0, device=pred_class_logits.device)
        if self.use_consistency and projected_3d_mask is not None and dino_2d_mask is not None:
            cons_loss = self.consistency_loss(projected_3d_mask, dino_2d_mask)
        
        # Total loss
        total_loss = (
            self.classification_weight * cls_loss +
            self.volume_weight * vol_loss +
            (self.consistency_weight * cons_loss if self.use_consistency else 0.0)
        )
        
        return {
            'total': total_loss,
            'classification': cls_loss,
            'volume': vol_loss,
            'consistency': cons_loss
        }

