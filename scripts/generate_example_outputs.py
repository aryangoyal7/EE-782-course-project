#!/usr/bin/env python3
"""
Script to generate example output visualizations for the README.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.inference.visualize import (
    visualize_voxels,
    plot_volume_scatter,
    plot_embedding_pca
)


def generate_example_outputs():
    """Generate example visualizations."""
    output_dir = Path("examples/expected_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating example outputs...")
    
    # Generate sample voxel grid
    voxel_resolution = 64
    voxels = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution))
    
    # Create a simple sphere-like shape
    center = voxel_resolution // 2
    radius = voxel_resolution // 3
    for i in range(voxel_resolution):
        for j in range(voxel_resolution):
            for k in range(voxel_resolution):
                dist = np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)
                if dist < radius:
                    voxels[i, j, k] = 1.0
    
    # Visualize voxels
    print("Generating voxel visualization...")
    visualize_voxels(voxels, str(output_dir / "predicted_voxels.png"))
    
    # Generate volume scatter plot
    print("Generating volume scatter plot...")
    np.random.seed(42)
    target_volumes = np.random.uniform(1000, 10000, 50)
    pred_volumes = target_volumes * (1 + np.random.normal(0, 0.1, 50))
    plot_volume_scatter(
        pred_volumes,
        target_volumes,
        str(output_dir / "volume_scatter.png")
    )
    
    # Generate PCA embedding visualization
    print("Generating PCA embedding...")
    np.random.seed(42)
    features = np.random.randn(100, 256)
    labels = np.random.randint(0, 10, 100)
    plot_embedding_pca(
        features,
        labels,
        str(output_dir / "pca_embedding.png"),
        title="Example Feature Embeddings (PCA)"
    )
    
    # Generate sample input image placeholder
    print("Generating sample input image...")
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title("Sample Input Image")
    plt.axis('off')
    plt.savefig(output_dir / "sample_input.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Example outputs generated in {output_dir}")


if __name__ == "__main__":
    generate_example_outputs()

