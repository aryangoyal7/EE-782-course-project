"""
I/O utilities for mesh, voxel, and image operations.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Tuple, Optional
from skimage import measure
import torch
from PIL import Image


def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    """
    Load a 3D mesh from file.
    
    Args:
        mesh_path: Path to mesh file (.obj, .ply, .stl, etc.)
    
    Returns:
        Trimesh object
    """
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        # If scene, get the first mesh
        mesh = list(mesh.geometry.values())[0]
    return mesh


def save_mesh(mesh: trimesh.Trimesh, output_path: str):
    """
    Save a 3D mesh to file.
    
    Args:
        mesh: Trimesh object
        output_path: Path to save mesh
    """
    mesh.export(output_path)


def mesh_to_voxels(
    mesh: trimesh.Trimesh,
    resolution: int = 64,
    padding: float = 0.1
) -> np.ndarray:
    """
    Convert mesh to voxel grid using ray casting.
    
    Args:
        mesh: Trimesh object
        resolution: Voxel grid resolution (NxNxN)
        padding: Padding factor around mesh bounds
    
    Returns:
        Binary voxel grid of shape (resolution, resolution, resolution)
    """
    # Get mesh bounds
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    size = np.max(bounds[1] - bounds[0]) * (1 + padding)
    
    # Create voxel grid
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    
    # Create grid points
    x = np.linspace(center[0] - size/2, center[0] + size/2, resolution)
    y = np.linspace(center[1] - size/2, center[1] + size/2, resolution)
    z = np.linspace(center[2] - size/2, center[2] + size/2, resolution)
    
    # Sample points and check if inside mesh
    points = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
    inside = mesh.contains(points)
    voxel_grid = inside.reshape(resolution, resolution, resolution).astype(np.float32)
    
    return voxel_grid


def voxels_to_mesh(
    voxels: np.ndarray,
    threshold: float = 0.5
) -> trimesh.Trimesh:
    """
    Convert voxel grid to mesh using marching cubes.
    
    Args:
        voxels: Voxel grid of shape (H, W, D) with values in [0, 1]
        threshold: Threshold for binarization
    
    Returns:
        Trimesh object
    """
    # Binarize voxels
    binary_voxels = (voxels > threshold).astype(np.float32)
    
    # Use marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        binary_voxels, level=0.5, spacing=(1.0, 1.0, 1.0)
    )
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    return mesh


def load_image(image_path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load and optionally resize an image.
    
    Args:
        image_path: Path to image file
        size: Optional (height, width) tuple for resizing
    
    Returns:
        Image array of shape (H, W, 3) with values in [0, 255]
    """
    img = Image.open(image_path).convert('RGB')
    if size is not None:
        img = img.resize((size[1], size[0]), Image.Resampling.LANCZOS)
    return np.array(img)


def save_image(image: np.ndarray, output_path: str):
    """
    Save image array to file.
    
    Args:
        image: Image array of shape (H, W, 3) with values in [0, 255]
        output_path: Path to save image
    """
    img = Image.fromarray(image.astype(np.uint8))
    img.save(output_path)


def load_voxels(voxel_path: str) -> np.ndarray:
    """
    Load voxel grid from .npy file.
    
    Args:
        voxel_path: Path to .npy file
    
    Returns:
        Voxel grid array
    """
    return np.load(voxel_path)


def save_voxels(voxels: np.ndarray, output_path: str):
    """
    Save voxel grid to .npy file.
    
    Args:
        voxels: Voxel grid array
        output_path: Path to save .npy file
    """
    np.save(output_path, voxels)

