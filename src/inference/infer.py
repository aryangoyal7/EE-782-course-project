"""
End-to-end inference script: images -> SAM-3D -> DINO -> fusion -> outputs.
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json

from src.utils.config import load_config
from src.models.dinov3_wrapper import DINOv3Wrapper
from src.models.sam3d_wrapper import SAM3DWrapper
from src.models.fusion_mlp import FusionMLP
from src.utils.io import save_voxels, voxels_to_mesh, save_mesh


def load_image(image_path: str, size: tuple = (224, 224)) -> torch.Tensor:
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size[1], size[0]), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return img_tensor


def infer_single(
    image_paths: list,
    dinov3: DINOv3Wrapper,
    sam3d: SAM3DWrapper,
    fusion_model: FusionMLP,
    device: torch.device,
    voxel_resolution: int = 64
) -> dict:
    """
    Run inference on a single sample.
    
    Args:
        image_paths: List of paths to input images
        dinov3: DINOv3 wrapper
        sam3d: SAM-3D wrapper
        fusion_model: Fusion MLP model
        device: Device to run on
        voxel_resolution: Resolution of voxel grid
    
    Returns:
        Dictionary with predictions
    """
    # Load images
    images = []
    for img_path in image_paths:
        img_tensor = load_image(img_path)
        images.append(img_tensor)
    
    # Stack images: (N_views, 1, 3, H, W) -> (1, N_views, 3, H, W)
    images = torch.cat(images, dim=0).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        # DINOv3 features
        _, semantic_features = dinov3(images)  # (1, N_views, D)
        semantic_features = semantic_features.mean(dim=1)  # (1, D)
        
        # SAM-3D features
        geometry_features, mask_logits = sam3d(images)  # (1, D_geom), (1, N_voxels)
    
    # Fusion model prediction
    with torch.no_grad():
        class_logits, volume_pred, voxel_logits = fusion_model(
            geometry_features, semantic_features
        )
        
        # Get predictions
        pred_class = class_logits.argmax(dim=1).item()
        pred_volume = volume_pred.item()
        
        # Reshape voxel predictions
        pred_voxels = torch.sigmoid(voxel_logits).view(
            1, voxel_resolution, voxel_resolution, voxel_resolution
        ).squeeze(0).cpu().numpy()
    
    return {
        'class': pred_class,
        'volume': pred_volume,
        'voxels': pred_voxels,
        'geometry_features': geometry_features.cpu().numpy(),
        'semantic_features': semantic_features.cpu().numpy()
    }


def main():
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to fusion model checkpoint")
    parser.add_argument("--images", type=str, nargs='+', required=True,
                       help="Paths to input images")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Directory to save outputs")
    parser.add_argument("--save_mesh", action="store_true",
                       help="Save predicted mesh")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
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
    print("Models loaded!")
    
    # Run inference
    print(f"Running inference on {len(args.images)} image(s)...")
    results = infer_single(
        args.images,
        dinov3, sam3d, fusion_model,
        device,
        voxel_resolution=config.data.voxel_resolution
    )
    
    # Save outputs
    print("Saving outputs...")
    
    # Save voxels
    voxel_path = os.path.join(args.output_dir, "predicted_voxels.npy")
    save_voxels(results['voxels'], voxel_path)
    print(f"Saved voxels to {voxel_path}")
    
    # Save mesh if requested
    if args.save_mesh:
        mesh = voxels_to_mesh(results['voxels'])
        mesh_path = os.path.join(args.output_dir, "predicted_mesh.obj")
        save_mesh(mesh, mesh_path)
        print(f"Saved mesh to {mesh_path}")
    
    # Save predictions JSON
    predictions = {
        'predicted_class': int(results['class']),
        'predicted_volume': float(results['volume']),
        'voxel_shape': list(results['voxels'].shape)
    }
    json_path = os.path.join(args.output_dir, "predictions.json")
    with open(json_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {json_path}")
    
    print("\nInference completed!")
    print(f"Predicted class: {results['class']}")
    print(f"Predicted volume: {results['volume']:.2f}")


if __name__ == "__main__":
    main()

