"""
Visualization utilities for 3D meshes, volume scatter plots, and embeddings.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple
import trimesh
import io
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.utils.io import load_mesh, voxels_to_mesh, load_image


def visualize_mesh(
    mesh: trimesh.Trimesh,
    output_path: str,
    resolution: Tuple[int, int] = (800, 600),
    camera_angle: Tuple[float, float] = (45, 45)
):
    """
    Render and save a 3D mesh visualization.
    
    Args:
        mesh: Trimesh object
        output_path: Path to save image
        resolution: (width, height) of output image
        camera_angle: (azimuth, elevation) in degrees
    """
    scene = mesh.scene()
    
    # Set camera
    camera_transform = trimesh.transformations.euler_matrix(
        np.radians(camera_angle[0]),
        np.radians(camera_angle[1]),
        0
    )
    scene.camera_transform = camera_transform
    
    # Render
    png = scene.save_image(resolution=resolution)
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(png)


def visualize_voxels(
    voxels: np.ndarray,
    output_path: str,
    threshold: float = 0.5
):
    """
    Visualize voxel grid as 3D plot.
    
    Args:
        voxels: Voxel grid (H, W, D)
        output_path: Path to save image
        threshold: Threshold for binarization
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Binarize
    binary_voxels = voxels > threshold
    
    # Get occupied voxel coordinates
    coords = np.where(binary_voxels)
    
    if len(coords[0]) > 0:
        ax.scatter(coords[0], coords[1], coords[2], c='blue', alpha=0.6, s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Voxel Grid Visualization')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_volume_scatter(
    pred_volumes: np.ndarray,
    target_volumes: np.ndarray,
    output_path: str,
    title: str = "Volume Prediction Scatter Plot"
):
    """
    Create scatter plot of predicted vs ground truth volumes.
    
    Args:
        pred_volumes: Predicted volumes (N,)
        target_volumes: Ground truth volumes (N,)
        output_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(target_volumes, pred_volumes, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_vol = min(target_volumes.min(), pred_volumes.min())
    max_vol = max(target_volumes.max(), pred_volumes.max())
    plt.plot([min_vol, max_vol], [min_vol, max_vol], 'r--', label='Perfect prediction')
    
    plt.xlabel('Ground Truth Volume')
    plt.ylabel('Predicted Volume')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(target_volumes, pred_volumes)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_embedding_pca(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_path: str = "pca_embedding.png",
    n_components: int = 2,
    title: str = "PCA Embedding Visualization"
):
    """
    Visualize features using PCA.
    
    Args:
        features: Feature vectors (N, D)
        labels: Optional class labels (N,)
        output_path: Path to save plot
        n_components: Number of PCA components
        title: Plot title
    """
    # Apply PCA
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        # Color by class
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                       c=[colors[i]], label=f'Class {label}', alpha=0.6, s=50)
        plt.legend()
    else:
        plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=50)
    
    plt.xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_embedding_tsne(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_path: str = "tsne_embedding.png",
    perplexity: float = 30.0,
    title: str = "t-SNE Embedding Visualization"
):
    """
    Visualize features using t-SNE.
    
    Args:
        features: Feature vectors (N, D)
        labels: Optional class labels (N,)
        output_path: Path to save plot
        perplexity: t-SNE perplexity parameter
        title: Plot title
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        # Color by class
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                       c=[colors[i]], label=f'Class {label}', alpha=0.6, s=50)
        plt.legend()
    else:
        plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=50)
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_visualization(
    pred_mesh_path: str,
    gt_mesh_path: str,
    output_path: str
):
    """
    Create side-by-side comparison of predicted and ground truth meshes.
    
    Args:
        pred_mesh_path: Path to predicted mesh
        gt_mesh_path: Path to ground truth mesh
        output_path: Path to save comparison image
    """
    pred_mesh = load_mesh(pred_mesh_path)
    gt_mesh = load_mesh(gt_mesh_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Render predicted mesh
    try:
        pred_scene = pred_mesh.scene()
        pred_png = pred_scene.save_image(resolution=(400, 400))
        pred_img = plt.imread(io.BytesIO(pred_png))
        axes[0].imshow(pred_img)
        axes[0].set_title('Predicted Mesh')
        axes[0].axis('off')
    except:
        axes[0].text(0.5, 0.5, 'Failed to render', ha='center', va='center')
        axes[0].set_title('Predicted Mesh')
    
    # Render ground truth mesh
    try:
        gt_scene = gt_mesh.scene()
        gt_png = gt_scene.save_image(resolution=(400, 400))
        gt_img = plt.imread(io.BytesIO(gt_png))
        axes[1].imshow(gt_img)
        axes[1].set_title('Ground Truth Mesh')
        axes[1].axis('off')
    except:
        axes[1].text(0.5, 0.5, 'Failed to render', ha='center', va='center')
        axes[1].set_title('Ground Truth Mesh')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

