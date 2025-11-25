"""
Tests for dataset loaders.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os
import json
from PIL import Image

from src.data.dataset import ShapeNetDataset, Pix3DDataset


def create_dummy_shapenet_dataset(tmpdir):
    """Create a dummy ShapeNet dataset structure."""
    data_root = Path(tmpdir) / "shapenet"
    
    # Create directory structure
    images_dir = data_root / "images" / "02691156" / "1a0e9c2e8c7b8e3f4a5b6c7d8e9f0a1b"
    meshes_dir = data_root / "meshes" / "02691156" / "1a0e9c2e8c7b8e3f4a5b6c7d8e9f0a1b"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy images
    for i in range(4):
        img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
        img.save(images_dir / f"render_{i:02d}.png")
    
    # Create dummy mesh (simple text file)
    mesh_path = meshes_dir / "model.obj"
    with open(mesh_path, 'w') as f:
        f.write("# Simple dummy mesh\n")
        f.write("v 0 0 0\n")
        f.write("v 1 0 0\n")
        f.write("v 0 1 0\n")
        f.write("f 1 2 3\n")
    
    return str(data_root)


def create_dummy_pix3d_dataset(tmpdir):
    """Create a dummy Pix3D dataset structure."""
    data_root = Path(tmpdir) / "pix3d"
    
    # Create directory structure
    img_dir = data_root / "img" / "chair"
    model_dir = data_root / "model" / "chair" / "sample1"
    
    img_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy image
    img = Image.new('RGB', (224, 224), color=(100, 100, 100))
    img.save(img_dir / "sample1.jpg")
    
    # Create dummy mesh
    mesh_path = model_dir / "model.obj"
    with open(mesh_path, 'w') as f:
        f.write("# Simple dummy mesh\n")
        f.write("v 0 0 0\n")
        f.write("v 1 0 0\n")
        f.write("v 0 1 0\n")
        f.write("f 1 2 3\n")
    
    return str(data_root)


def test_shapenet_dataset_creation():
    """Test ShapeNet dataset creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = create_dummy_shapenet_dataset(tmpdir)
        
        dataset = ShapeNetDataset(
            data_root=data_root,
            split="train",
            voxel_resolution=32,
            image_size=(224, 224),
            num_views=4
        )
        
        assert len(dataset) > 0


def test_shapenet_dataset_getitem():
    """Test ShapeNet dataset __getitem__."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = create_dummy_shapenet_dataset(tmpdir)
        
        dataset = ShapeNetDataset(
            data_root=data_root,
            split="train",
            voxel_resolution=32,
            image_size=(224, 224),
            num_views=4
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            
            assert 'images' in sample
            assert 'voxels' in sample
            assert 'class_label' in sample
            assert 'volume' in sample
            
            assert sample['images'].shape == (4, 3, 224, 224)
            assert sample['voxels'].shape == (32, 32, 32)
            assert isinstance(sample['class_label'], torch.Tensor)
            assert isinstance(sample['volume'], torch.Tensor)


def test_pix3d_dataset_creation():
    """Test Pix3D dataset creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = create_dummy_pix3d_dataset(tmpdir)
        
        dataset = Pix3DDataset(
            data_root=data_root,
            split="train",
            voxel_resolution=32,
            image_size=(224, 224)
        )
        
        assert len(dataset) > 0


def test_pix3d_dataset_getitem():
    """Test Pix3D dataset __getitem__."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = create_dummy_pix3d_dataset(tmpdir)
        
        dataset = Pix3DDataset(
            data_root=data_root,
            split="train",
            voxel_resolution=32,
            image_size=(224, 224)
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            
            assert 'images' in sample
            assert 'voxels' in sample
            assert 'class_label' in sample
            assert 'volume' in sample
            
            assert sample['images'].shape == (1, 3, 224, 224)
            assert sample['voxels'].shape == (32, 32, 32)

