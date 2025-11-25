"""
Training script for the fusion MLP model.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.utils.config import load_config
from src.data.dataset import ShapeNetDataset, Pix3DDataset
from src.models.dinov3_wrapper import DINOv3Wrapper
from src.models.sam3d_wrapper import SAM3DWrapper
from src.models.fusion_mlp import FusionMLP
from src.training.losses import CombinedLoss
from src.utils.metrics import evaluate_batch


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    dinov3: DINOv3Wrapper,
    sam3d: SAM3DWrapper,
    dataloader: DataLoader,
    criterion: CombinedLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_consistency: bool = True
) -> dict:
    """Train for one epoch."""
    model.train()
    dinov3.eval()  # Freeze pretrained models
    sam3d.eval()
    
    total_loss = 0.0
    loss_components = {'classification': 0.0, 'volume': 0.0, 'consistency': 0.0}
    
    with torch.no_grad():
        # Extract features once (pretrained models frozen)
        pass
    
    for batch in tqdm(dataloader, desc="Training"):
        images = batch['images'].to(device)  # (B, N_views, C, H, W)
        target_voxels = batch['voxels'].to(device)  # (B, H, W, D)
        target_class = batch['class_label'].to(device)  # (B,)
        target_volume = batch['volume'].to(device)  # (B,)
        
        # Extract features
        with torch.no_grad():
            # DINOv3 features
            _, semantic_features = dinov3(images)  # (B, N_views, D)
            semantic_features = semantic_features.mean(dim=1)  # (B, D) - average over views
            
            # SAM-3D features
            geometry_features, _ = sam3d(images)  # (B, D_geom)
        
        # Forward pass through fusion model
        optimizer.zero_grad()
        
        class_logits, volume_pred, voxel_logits = model(
            geometry_features, semantic_features
        )
        
        # Compute consistency loss if needed
        projected_3d_mask = None
        dino_2d_mask = None
        if use_consistency:
            # Reshape voxel logits
            voxel_resolution = int(voxel_logits.shape[1] ** (1/3))
            voxel_pred = voxel_logits.view(-1, voxel_resolution, voxel_resolution, voxel_resolution)
            
            # Project to 2D (simplified)
            from src.models.fusion_mlp import ConsistencyProjection
            projector = ConsistencyProjection(voxel_resolution=voxel_resolution).to(device)
            projected_3d_mask = projector(voxel_pred)
            
            # Get 2D mask from DINO (simplified: use patch attention)
            # In practice, this would use a learned 2D mask predictor
            dino_2d_mask = torch.sigmoid(semantic_features[:, :224*224].view(-1, 224, 224))
        
        # Compute loss
        loss_dict = criterion(
            class_logits, target_class,
            volume_pred, target_volume,
            projected_3d_mask, dino_2d_mask
        )
        
        loss = loss_dict['total']
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key].item()
    
    num_batches = len(dataloader)
    return {
        'total': total_loss / num_batches,
        **{k: v / num_batches for k, v in loss_components.items()}
    }


def validate(
    model: nn.Module,
    dinov3: DINOv3Wrapper,
    sam3d: SAM3DWrapper,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """Validate model."""
    model.eval()
    dinov3.eval()
    sam3d.eval()
    
    all_pred_classes = []
    all_target_classes = []
    all_pred_volumes = []
    all_target_volumes = []
    all_pred_voxels = []
    all_target_voxels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['images'].to(device)
            target_voxels = batch['voxels'].to(device)
            target_class = batch['class_label'].to(device)
            target_volume = batch['volume'].to(device)
            
            # Extract features
            _, semantic_features = dinov3(images)
            semantic_features = semantic_features.mean(dim=1)
            geometry_features, _ = sam3d(images)
            
            # Forward pass
            class_logits, volume_pred, voxel_logits = model(
                geometry_features, semantic_features
            )
            
            # Predictions
            pred_class = class_logits.argmax(dim=1)
            pred_volume = volume_pred.squeeze(1)
            
            # Reshape voxel predictions
            voxel_resolution = int(voxel_logits.shape[1] ** (1/3))
            pred_voxels = torch.sigmoid(voxel_logits).view(
                -1, voxel_resolution, voxel_resolution, voxel_resolution
            )
            
            # Collect
            all_pred_classes.append(pred_class.cpu())
            all_target_classes.append(target_class.cpu())
            all_pred_volumes.append(pred_volume.cpu())
            all_target_volumes.append(target_volume.cpu())
            all_pred_voxels.append(pred_voxels.cpu())
            all_target_voxels.append(target_voxels.cpu())
    
    # Concatenate and compute metrics
    pred_classes = torch.cat(all_pred_classes, dim=0)
    target_classes = torch.cat(all_target_classes, dim=0)
    pred_volumes = torch.cat(all_pred_volumes, dim=0)
    target_volumes = torch.cat(all_target_volumes, dim=0)
    pred_voxels = torch.cat(all_pred_voxels, dim=0)
    target_voxels = torch.cat(all_target_voxels, dim=0)
    
    metrics = evaluate_batch(
        pred_voxels, target_voxels,
        pred_classes, target_classes,
        pred_volumes, target_volumes
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train fusion MLP model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config.training.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.training.save_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)
    
    # Load dataset
    if config.data.dataset_name == "shapenet":
        train_dataset = ShapeNetDataset(
            data_root=config.data.data_root,
            split="train",
            voxel_resolution=config.data.voxel_resolution,
            image_size=config.data.image_size,
            num_views=config.data.num_views
        )
        val_dataset = ShapeNetDataset(
            data_root=config.data.data_root,
            split="val",
            voxel_resolution=config.data.voxel_resolution,
            image_size=config.data.image_size,
            num_views=config.data.num_views
        )
    else:  # pix3d
        train_dataset = Pix3DDataset(
            data_root=config.data.data_root,
            split="train",
            voxel_resolution=config.data.voxel_resolution,
            image_size=config.data.image_size
        )
        val_dataset = Pix3DDataset(
            data_root=config.data.data_root,
            split="val",
            voxel_resolution=config.data.voxel_resolution,
            image_size=config.data.image_size
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Load models
    dinov3 = DINOv3Wrapper(
        model_name=config.model.dinov3_model,
        pretrained_path=config.model.dinov3_pretrained_path if config.model.dinov3_pretrained_path else None
    ).to(device)
    dinov3.eval()
    
    sam3d = SAM3DWrapper(
        checkpoint_path=config.model.sam3d_checkpoint_path if config.model.sam3d_checkpoint_path else None,
        model_type=config.model.sam3d_model_type
    ).to(device)
    sam3d.eval()
    
    fusion_model = FusionMLP(
        geometry_dim=sam3d.feature_dim,
        semantic_dim=dinov3.feature_dim,
        hidden_dims=config.model.fusion_hidden_dims,
        num_classes=config.model.num_classes,
        dropout=config.model.fusion_dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(use_consistency=config.model.use_consistency_loss)
    
    if config.training.optimizer == "adam":
        optimizer = optim.Adam(
            fusion_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    else:
        optimizer = optim.SGD(
            fusion_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            momentum=0.9
        )
    
    # Scheduler
    if config.training.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.num_epochs
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        fusion_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_val_iou = 0.0
    history = {'train_loss': [], 'val_metrics': []}
    
    for epoch in range(start_epoch, config.training.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.training.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            fusion_model, dinov3, sam3d, train_loader,
            criterion, optimizer, device,
            use_consistency=config.model.use_consistency_loss
        )
        history['train_loss'].append(train_loss)
        
        # Validate
        if (epoch + 1) % config.training.eval_freq == 0:
            val_metrics = validate(fusion_model, dinov3, sam3d, val_loader, device)
            history['val_metrics'].append(val_metrics)
            print(f"Validation - IoU: {val_metrics['iou']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f}, "
                  f"MAPE: {val_metrics['mape']:.2f}%")
            
            # Save best model
            if val_metrics['iou'] > best_val_iou:
                best_val_iou = val_metrics['iou']
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': fusion_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_iou': val_metrics['iou'],
                }, os.path.join(config.training.save_dir, 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % config.training.save_freq == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(config.training.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        scheduler.step()
    
    # Save training history
    with open(os.path.join(config.training.log_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

