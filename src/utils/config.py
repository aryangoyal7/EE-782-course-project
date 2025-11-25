"""
Configuration management for the 3D object understanding pipeline.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for model components."""
    # DINOv3 settings
    dinov3_model: str = "dinov2_vitb14"
    dinov3_pretrained_path: str = ""
    dinov3_patch_size: int = 14
    
    # SAM-3D settings
    sam3d_checkpoint_path: str = ""
    sam3d_model_type: str = "sam3d_vit_h"
    
    # Fusion MLP settings
    fusion_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    fusion_dropout: float = 0.1
    num_classes: int = 10
    use_consistency_loss: bool = True


@dataclass
class DataConfig:
    """Configuration for data loading."""
    dataset_name: str = "shapenet"  # or "pix3d"
    data_root: str = "./data"
    voxel_resolution: int = 64
    image_size: tuple = (224, 224)
    num_views: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_freq: int = 5
    eval_freq: int = 1
    seed: int = 42


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__
        }


def load_config(config_path: str = None) -> Config:
    """Load configuration from file or return default."""
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    return Config()

