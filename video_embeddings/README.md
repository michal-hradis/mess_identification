# Video Embeddings Implementation

Complete implementation of JEPA-like video embedding learning with PyTorch Lightning.

## Overview

This implementation provides:
- **Video dataset** for sampling frames from LMDB or directory
- **JEPA model** with student-teacher architecture and EMA updates
- **PyTorch Lightning** integration for multi-GPU training
- **GPU augmentation** support using Kornia
- **TensorBoard and ClearML** logging
- **Export utilities** for trained models

## Files

### Core Implementation
- `video_dataset.py` - VideoDataset for loading video frames
- `jepa_model.py` - FrameEncoder, TemporalPredictor, and JEPAVideoModel
- `lightning_module.py` - PyTorch Lightning wrapper (VideoEmbeddingModule)
- `train_video_embeddings.py` - Main training script
- `utils.py` - Export and inference utilities
- `__init__.py` - Module initialization

### Testing and Documentation
- `test_components.py` - Unit tests for model components
- `video_embeddings.md` - Detailed documentation
- `example_train.sh` - Example training commands

## Installation

### Required Dependencies
```bash
pip install torch torchvision
pip install pytorch-lightning
pip install lmdb
pip install opencv-python
pip install segmentation-models-pytorch  # For SM encoders
pip install timm  # For TIMM encoders (optional)
pip install kornia  # For GPU augmentation (optional)
pip install clearml  # For experiment tracking (optional)
```

## Quick Start

### 1. Prepare Data

Your data should be in LMDB or a directory with frames named as:
```
{video_id}_{frame_id}.jpg
```

Example:
```
video001_00001.jpg
video001_00002.jpg
video002_00001.jpg
...
```

### 2. Train a Model

Basic training:
```bash
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --name my_video_model \
    --encoder-config '{"type":"sm","name":"resnet34","weights":"imagenet","depth":5}' \
    --embedding-dim 256 \
    --num-frames 8 \
    --batch-size 16 \
    --gpus 1
```

With validation:
```bash
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --val-data-path /path/to/val.lmdb \
    --name my_video_model \
    --gpus 1
```

Multi-GPU training:
```bash
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --name my_video_model \
    --gpus 4 \
    --batch-size 8 \
    --accumulate-grad-batches 2
```

### 3. Export Trained Model

```bash
python utils.py \
    --checkpoint lightning_logs/my_video_model/checkpoints/last.ckpt \
    --output encoder.pt
```

### 4. Use for Inference

```python
from video_embeddings.utils import load_encoder_for_inference, extract_video_embedding
import torch

# Load encoder
encoder = load_encoder_for_inference('encoder.pt', device='cuda')

# Extract embedding from video frames
frames = torch.randint(0, 256, (16, 3, 224, 224), dtype=torch.uint8).cuda()
embedding = extract_video_embedding(encoder, frames, aggregate='mean')
```

## Configuration

### Encoder Configurations

**Segmentation Models PyTorch (recommended):**
```json
{"type":"sm","name":"resnet34","weights":"imagenet","depth":5}
{"type":"sm","name":"resnet50","weights":"imagenet","depth":5}
{"type":"sm","name":"efficientnet-b0","weights":"imagenet","depth":5}
```

**TIMM:**
```json
{"type":"timm","name":"resnet34","pretrained":true}
{"type":"timm","name":"efficientnet_b0","pretrained":true}
```

**TorchVision:**
```json
{"type":"torchvision","name":"resnet34","weights":"DEFAULT"}
{"type":"torchvision","name":"mobilenet_v3_large","weights":"DEFAULT"}
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | Required | Path to training data (LMDB or directory) |
| `--val-data-path` | None | Path to validation data (optional) |
| `--num-frames` | 8 | Number of frames to sample per video |
| `--max-frame-gap` | 5 | Maximum gap between consecutive frames |
| `--image-size` | 224 224 | Image size (height width) |
| `--embedding-dim` | 256 | Embedding dimension |
| `--context-frames` | 4 | Number of context frames for prediction |
| `--batch-size` | 16 | Batch size per GPU |
| `--learning-rate` | 1e-4 | Learning rate |
| `--weight-decay` | 0.01 | Weight decay |
| `--warmup-steps` | 1000 | Number of warmup steps |
| `--max-steps` | 100000 | Maximum training steps |
| `--gpus` | 1 | Number of GPUs |
| `--precision` | 16-mixed | Training precision (32, 16-mixed, bf16-mixed) |
| `--gpu-augmentation` | None | GPU augmentation pipeline (aug_1) |

## Model Architecture

### FrameEncoder
- Wraps any image encoder (SM, TIMM, or TorchVision)
- Projects features to embedding dimension
- Optional L2 normalization

### TemporalPredictor
- Transformer-based predictor
- Learnable temporal position embeddings
- Predicts future frame embeddings from past frames

### JEPAVideoModel
- Student-teacher architecture
- EMA updates for teacher encoder
- JEPA-like self-supervised learning

## Loss Functions

Available loss types:
- `smooth_l1` - Smooth L1 loss (default, robust to outliers)
- `mse` - Mean squared error
- `cosine` - Cosine similarity loss

## GPU Augmentation

Enable with `--gpu-augmentation aug_1`:
- Random horizontal/vertical flip
- Random affine transformations
- Color jitter
- Random erasing
- Random grayscale

## Logging

### TensorBoard (default)
```bash
tensorboard --logdir lightning_logs
```

### ClearML (optional)
```bash
python train_video_embeddings.py \
    --use-clearml \
    --project "VideoEmbeddings" \
    --name my_experiment
```

## Testing

Run component tests:
```bash
cd video_embeddings
python test_components.py
```

## Advanced Usage

### Custom Dataset
```python
from video_embeddings import VideoDataset

dataset = VideoDataset(
    data_path='/path/to/data',
    num_frames=16,
    max_frame_gap=3,
    image_size=(256, 256),
    is_lmdb=True
)
```

### Video-level Encoder
```python
from video_embeddings import VideoEncoder

video_encoder = VideoEncoder(
    frame_encoder=encoder,
    aggregate='mean',  # or 'max', 'first', 'last'
    normalize=True
)

# Process batch of videos
videos = torch.randint(0, 256, (4, 16, 3, 224, 224), dtype=torch.uint8)
embeddings = video_encoder(videos)  # (4, 256)
```

### Resume Training
```bash
python train_video_embeddings.py \
    --resume-from lightning_logs/my_video_model/checkpoints/last.ckpt \
    --data-path /path/to/train.lmdb
```

## Performance Tips

1. **Batch size**: Use largest batch size that fits in GPU memory
2. **Gradient accumulation**: Use `--accumulate-grad-batches` for effective larger batches
3. **Mixed precision**: Use `--precision 16-mixed` or `bf16-mixed` for faster training
4. **Persistent workers**: Dataset uses persistent workers for faster data loading
5. **GPU augmentation**: Faster than CPU augmentation, enable with `--gpu-augmentation aug_1`

## Citation

If you use this implementation, please cite:
```
@software{video_embeddings_jepa,
  title={Video Embedding Learning with JEPA},
  author={Your Name},
  year={2026}
}
```

## License

See LICENSE file in the project root.
