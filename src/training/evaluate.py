"""
Evaluation script for the fusion MLP model.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np

from src.utils.config import load_config
from src.data.dataset import ShapeNetDataset, Pix3DDataset
from src.models.dinov3_wrapper import DINOv3Wrapper
from src.models.sam3d_wrapper import SAM3DWrapper
from src.models.fusion_mlp import FusionMLP
from src.utils.metrics import evaluate_batch


def evaluate(
    model: FusionMLP,
    dinov3: DINOv3Wrapper,
    sam3d: SAM3DWrapper,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str = None
) -> dict:
    """
    Evaluate model on dataset.
    
    Args:
        model: Fusion MLP model
        dinov3: DINOv3 wrapper
        sam3d: SAM-3D wrapper
        dataloader: Data loader
        device: Device to run on
        output_dir: Optional directory to save predictions
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    dinov3.eval()
    sam3d.eval()
    
    all_pred_classes = []
    all_target_classes = []
    all_pred_volumes = []
    all_target_volumes = []
    all_pred_voxels = []
    all_target_voxels = []
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
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
            
            # Save sample predictions
            if output_dir and batch_idx < 5:  # Save first 5 samples
                sample_dir = os.path.join(output_dir, f"sample_{batch_idx}")
                os.makedirs(sample_dir, exist_ok=True)
                
                # Save voxel predictions
                np.save(
                    os.path.join(sample_dir, "pred_voxels.npy"),
                    pred_voxels[0].numpy()
                )
                np.save(
                    os.path.join(sample_dir, "target_voxels.npy"),
                    target_voxels[0].cpu().numpy()
                )
    
    # Concatenate
    pred_classes = torch.cat(all_pred_classes, dim=0)
    target_classes = torch.cat(all_target_classes, dim=0)
    pred_volumes = torch.cat(all_pred_volumes, dim=0)
    target_volumes = torch.cat(all_target_volumes, dim=0)
    pred_voxels = torch.cat(all_pred_voxels, dim=0)
    target_voxels = torch.cat(all_target_voxels, dim=0)
    
    # Compute metrics
    metrics = evaluate_batch(
        pred_voxels, target_voxels,
        pred_classes, target_classes,
        pred_volumes, target_volumes
    )
    
    # Per-class metrics
    unique_classes = torch.unique(target_classes)
    per_class_metrics = {}
    for cls in unique_classes:
        mask = target_classes == cls
        if mask.sum() > 0:
            cls_iou = evaluate_batch(
                pred_voxels[mask], target_voxels[mask],
                pred_classes[mask], target_classes[mask],
                pred_volumes[mask], target_volumes[mask]
            )
            per_class_metrics[int(cls)] = cls_iou
    
    metrics['per_class'] = per_class_metrics
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate fusion MLP model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Dataset split to evaluate on")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save predictions")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    if config.data.dataset_name == "shapenet":
        dataset = ShapeNetDataset(
            data_root=config.data.data_root,
            split=args.split,
            voxel_resolution=config.data.voxel_resolution,
            image_size=config.data.image_size,
            num_views=config.data.num_views
        )
    else:  # pix3d
        dataset = Pix3DDataset(
            data_root=config.data.data_root,
            split=args.split,
            voxel_resolution=config.data.voxel_resolution,
            image_size=config.data.image_size
        )
    
    dataloader = DataLoader(
        dataset, batch_size=config.training.batch_size,
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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    fusion_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Evaluate
    metrics = evaluate(
        fusion_model, dinov3, sam3d, dataloader, device,
        output_dir=args.output_dir
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Volume MAPE: {metrics['mape']:.2f}%")
    print("\nPer-class metrics:")
    for cls, cls_metrics in metrics['per_class'].items():
        print(f"  Class {cls}: IoU={cls_metrics['iou']:.4f}, "
              f"Acc={cls_metrics['accuracy']:.4f}")
    
    # Save metrics
    if args.output_dir:
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

