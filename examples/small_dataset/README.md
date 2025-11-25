# Small Example Dataset

This directory contains a small synthetic dataset for quick testing.

## Structure

```
small_dataset/
  images/
    sample1/
      render_00.png
      render_01.png
      render_02.png
      render_03.png
  meshes/
    sample1/
      model.obj
```

## Usage

To use this dataset, update your config file:

```yaml
data:
  data_root: "./examples/small_dataset"
  dataset_name: "shapenet"
```

Or use it directly in code:

```python
from src.data.dataset import ShapeNetDataset

dataset = ShapeNetDataset(
    data_root="./examples/small_dataset",
    split="train",
    voxel_resolution=32,
    num_views=4
)
```

## Generating Synthetic Data

You can generate synthetic rendered views using the `render_utils` module:

```python
from src.data.render_utils import render_mesh_views

render_mesh_views(
    mesh_path="path/to/mesh.obj",
    output_dir="output/images",
    num_views=4
)
```

