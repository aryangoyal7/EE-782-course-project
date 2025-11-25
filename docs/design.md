# Design Document

## Architecture Overview

This project implements a 3D object understanding pipeline that combines geometry and semantic features from multiple pretrained models:

1. **SAM-3D**: Extracts 3D geometry features and mask predictions
2. **DINOv3**: Extracts semantic features from multi-view images
3. **Fusion MLP**: Combines features to predict object class and volume

## Key Design Decisions

### Feature Extraction

- **DINOv3**: We use DINOv2 (DINOv3) Vision Transformer models pretrained on large-scale image data. The model extracts patch tokens and pooled embeddings from input images. For multi-view inputs, we average the pooled embeddings across views.

- **SAM-3D**: We use SAM-3D (Segment Anything Model 3D) to extract 3D geometry features. The model processes multi-view images and outputs geometry features and 3D mask logits. Currently, we use a placeholder implementation that simulates the expected outputs.

### Fusion Architecture

The Fusion MLP takes concatenated geometry and semantic features and processes them through shared layers before branching into three heads:

1. **Classification Head**: Predicts object class (CrossEntropy loss)
2. **Volume Regression Head**: Predicts object volume (SmoothL1 loss)
3. **Voxel Prediction Head**: Predicts full 3D voxel grid (for consistency loss)

### Consistency Loss

We implement an optional consistency loss that penalizes mismatch between:
- Projected 3D mask (from voxel predictions)
- 2D mask predicted from DINO features

This encourages geometric and semantic consistency across 2D and 3D representations.

### Data Processing

- **Mesh to Voxel**: We use trimesh to load meshes and convert them to voxel grids using ray casting
- **Voxel to Mesh**: We use marching cubes (scikit-image) to convert voxel grids back to meshes
- **Multi-view Rendering**: We provide utilities to render synthetic views using pyrender

## Implementation Details

### Modularity

The codebase is organized into clear modules:
- `src/data/`: Dataset loaders and rendering utilities
- `src/models/`: Model definitions and wrappers
- `src/training/`: Training and evaluation scripts
- `src/inference/`: Inference and visualization
- `src/utils/`: Configuration, metrics, and I/O utilities

### Extensibility

- Model wrappers use a clear interface, making it easy to swap in actual pretrained checkpoints
- Configuration is managed through YAML files
- Loss functions are modular and can be easily combined

### Reproducibility

- Random seeds are set for deterministic training
- Configuration files capture all hyperparameters
- Training history is saved as JSON

## Future Improvements

1. **Actual SAM-3D Integration**: Replace placeholder with real SAM-3D checkpoint loading
2. **Better 2D-3D Consistency**: Implement proper camera projection for consistency loss
3. **Multi-scale Features**: Use features from multiple DINOv3 layers
4. **Attention Mechanisms**: Add attention to fuse multi-view features more effectively
5. **Data Augmentation**: Add robust augmentation strategies

