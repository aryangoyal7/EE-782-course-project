# 3D Object Understanding Pipeline

A complete pipeline for 3D object understanding that combines geometry features from SAM-3D and semantic features from DINOv3 through a fusion MLP to predict object class and volume.

## Abstract

This project implements an end-to-end pipeline for 3D object understanding from multi-view images. The approach combines:

- **SAM-3D (Segment Anything Model 3D)**: Extracts 3D geometry features and mask predictions from multi-view images
- **DINOv3**: Extracts rich semantic features using self-supervised Vision Transformers
- **Fusion MLP**: Combines geometry and semantic features to predict object class and volume

The pipeline supports both ShapeNet rendered views and Pix3D single-image scenarios, with comprehensive training, evaluation, and visualization tools.

## Quickstart

### Requirements

#### Using Conda (Recommended)

```bash
conda create -n 3d-understanding python=3.9
conda activate 3d-understanding
pip install -r requirements.txt
```

#### Using Pip

```bash
pip install -r requirements.txt
```

#### Using Docker

```bash
docker build -t 3d-understanding .
docker run -it --gpus all -v $(pwd):/workspace 3d-understanding
```

### Running the Tiny Example

1. **Clone the repository**:
```bash
git clone <repository-url>
cd EE-782-course-project
```

2. **Install dependencies** (see Requirements above)

3. **Download pretrained models** (optional, will use timm pretrained weights if not provided):
   - DINOv2: Automatically downloaded via timm
   - SAM-3D: Place checkpoint in `pretrained/sam3d/` (see instructions below)

4. **Run inference on example images**:
```bash
python -m src.inference.infer \
    --checkpoint checkpoints/best_model.pth \
    --images examples/small_dataset/images/sample1/render_00.png \
    --output_dir outputs \
    --save_mesh
```

5. **Run training** (if you have data):
```bash
bash scripts/run_train.sh --config configs/default.yaml
```

## Datasets

### ShapeNet

ShapeNet provides multi-view rendered images of 3D models.

**Downloading ShapeNet Renders**:
1. Visit [ShapeNet](https://www.shapenet.org/)
2. Download ShapeNetCore.v2
3. Use rendering scripts to generate multi-view images, or use provided `render_utils.py`

**Expected Structure**:
```
data/shapenet/
  images/
    {synset_id}/{model_id}/render_{view_id}.png
  meshes/
    {synset_id}/{model_id}/model.obj
  metadata.json  # Auto-generated if not present
```

**Generating Renders**:
```python
from src.data.render_utils import create_synthetic_dataset

create_synthetic_dataset(
    mesh_dir="path/to/shapenet/meshes",
    output_dir="data/shapenet/images",
    num_views=4
)
```

### Pix3D

Pix3D provides single images with corresponding 3D models.

**Downloading Pix3D**:
1. Visit [Pix3D](https://github.com/xingyuansun/pix3d)
2. Download the dataset
3. Organize according to expected structure

**Expected Structure**:
```
data/pix3d/
  img/
    {category}/{model_id}.jpg
  model/
    {category}/{model_id}/model.obj
  metadata.json  # Auto-generated if not present
```

### Small Example Dataset

A tiny synthetic dataset is included in `examples/small_dataset/` for quick testing. See `examples/small_dataset/README.md` for details.

## Usage

### Training

Train the fusion MLP model:

```bash
# Using default config
python -m src.training.train --config configs/default.yaml

# Using custom config
python -m src.training.train --config configs/shapenet_experiment.yaml

# Resume from checkpoint
python -m src.training.train --config configs/default.yaml --resume checkpoints/checkpoint_epoch_10.pth

# Or use the script
bash scripts/run_train.sh --config configs/default.yaml
```

**Training Outputs**:
- Model checkpoints saved to `checkpoints/` (or path specified in config)
- Training logs saved to `logs/` (or path specified in config)
- Best model saved as `checkpoints/best_model.pth`

### Inference

Run inference on new images:

```bash
# Single image
python -m src.inference.infer \
    --checkpoint checkpoints/best_model.pth \
    --images path/to/image.png \
    --output_dir outputs

# Multiple images (multi-view)
python -m src.inference.infer \
    --checkpoint checkpoints/best_model.pth \
    --images img1.png img2.png img3.png img4.png \
    --output_dir outputs \
    --save_mesh

# Or use the script
bash scripts/run_infer.sh \
    --checkpoint checkpoints/best_model.pth \
    --images img1.png img2.png \
    --output_dir outputs
```

**Inference Outputs**:
- `predicted_voxels.npy`: Predicted voxel grid
- `predicted_mesh.obj`: Predicted 3D mesh (if `--save_mesh` is used)
- `predictions.json`: JSON with predicted class and volume

### Evaluation

Evaluate model performance:

```bash
python -m src.training.evaluate \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_model.pth \
    --split test \
    --output_dir evaluation_results
```

**Evaluation Metrics**:
- **IoU**: Intersection over Union for voxel predictions
- **Accuracy**: Classification accuracy
- **MAPE**: Mean Absolute Percentage Error for volume regression

Results are saved to `evaluation_results/metrics.json` and sample predictions are saved for visualization.

### Visualization

Generate visualizations:

```python
from src.inference.visualize import (
    visualize_voxels,
    plot_volume_scatter,
    plot_embedding_pca
)

# Visualize voxel grid
visualize_voxels(voxels, "output.png")

# Volume scatter plot
plot_volume_scatter(pred_volumes, target_volumes, "scatter.png")

# PCA embedding visualization
plot_embedding_pca(features, labels, "pca.png")
```

## Repository Structure

```
.
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── .gitignore               # Git ignore rules
│
├── src/                     # Source code
│   ├── data/
│   │   ├── dataset.py       # ShapeNet and Pix3D loaders
│   │   └── render_utils.py  # Mesh rendering utilities
│   ├── models/
│   │   ├── dinov3_wrapper.py    # DINOv3 feature extraction
│   │   ├── sam3d_wrapper.py     # SAM-3D wrapper (placeholder)
│   │   └── fusion_mlp.py         # Fusion MLP model
│   ├── training/
│   │   ├── train.py         # Training script
│   │   ├── evaluate.py      # Evaluation script
│   │   └── losses.py        # Loss functions
│   ├── inference/
│   │   ├── infer.py         # Inference script
│   │   └── visualize.py     # Visualization utilities
│   └── utils/
│       ├── config.py        # Configuration management
│       ├── metrics.py       # Evaluation metrics
│       └── io.py            # I/O utilities
│
├── notebooks/               # Jupyter notebooks
│   ├── demo_shapenet.ipynb
│   └── demo_pix3d.ipynb
│
├── configs/                 # Configuration files
│   ├── default.yaml
│   └── shapenet_experiment.yaml
│
├── tests/                   # Unit tests
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_inference.py
│
├── scripts/                 # Shell scripts
│   ├── run_train.sh
│   └── run_infer.sh
│
├── examples/                # Example data and outputs
│   ├── small_dataset/
│   └── expected_outputs/
│
└── docs/                    # Documentation
    ├── design.md
    └── citations.bib
```

## Reproducibility

### Deterministic Training

The code uses fixed random seeds for reproducibility:

- Random seed: 42 (configurable in config file)
- PyTorch deterministic mode enabled
- NumPy random seed set

### GPU Requirements

- **Training**: Recommended GPU with at least 8GB VRAM (e.g., NVIDIA RTX 2080, V100)
- **Inference**: Can run on CPU, but GPU recommended for faster processing
- **Docker**: Use `--gpus all` flag when running Docker container

### Environment Variables

Set the following for deterministic behavior:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
```

### Pretrained Models

**DINOv2/DINOv3**:
- Automatically downloaded via `timm` when first used
- Models: `dinov2_vitb14`, `dinov2_vitl14`, etc.
- No manual download required

**SAM-3D**:
- Currently uses placeholder implementation
- To use actual SAM-3D:
  1. Download checkpoint from [SAM-3D repository](https://github.com/Pointcept/SAM3D)
  2. Place in `pretrained/sam3d/`
  3. Update `sam3d_checkpoint_path` in config file
  4. Update `src/models/sam3d_wrapper.py` to load actual checkpoint

## Citation

If you use this code in your research, please cite:

```bibtex
@software{3d_understanding_pipeline,
  title={3D Object Understanding Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/EE-782-course-project}
}
```

### Key References

- **SAM**: [Segment Anything](https://arxiv.org/abs/2304.02643)
- **DINOv2**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- **SAM-3D**: [SAM3D: Segment Anything in 3D Scenes](https://arxiv.org/abs/2306.03908)
- **ShapeNet**: [3D ShapeNets: A deep representation for volumetric shapes](https://arxiv.org/abs/1406.5670)
- **Pix3D**: [Pix3D: Dataset and methods for single-image 3D shape modeling](https://arxiv.org/abs/1804.09686)
- **Point Transformer V3**: [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035)

See `docs/citations.bib` for full bibliography.




### Testing

Run tests with:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- DINOv2 team for the excellent self-supervised vision transformer
- SAM team for the Segment Anything Model
- ShapeNet and Pix3D dataset creators
- PyTorch community for excellent tools and documentation

