This directory contains code video embedding learning experiments. 
It supports ClearML integration for experiment tracking. 
Multi-GPU training is supported via PyTorch Lightning.

# Dataset
The dataset reads either a local directory or LMDB containing images extracted from videos. File names and LMDB keys have 
the format `{video_id}_{frame_id}.jpg`, where `video_id` is a unique identifier for each video and `frame_id` is the frame number within that video.

**Implementation**: `video_dataset.py` - `VideoDataset` class

### Parameters:
- `data_path`: Path to the directory or LMDB containing the images.
- `num_frames`: Number of frames to sample from each video.
- `max_frame_gap`: Maximum gap between consecutive frames when sampling.
- `image_size`: Tuple (H, W) for resizing frames.
- `is_lmdb`: Whether data_path is LMDB (True) or directory (False).

### Behavior:
The dataset returns an uint8 RGB tensor of shape `(num_frames, 3, H, W)` for each video, 
where `H` and `W` are the height and width of the images. The tensor contains consecutive frames sampled from a random 
starting point in the video.

# Models
The script supports various models:
- models from ./common/nets_pretrained.py (segmentation_models_pytorch encoders)
- timm models
- torchvision models

All of these models can be configured to specific embedding dimensions using a linear layer.

**Implementation**: 
- `jepa_model.py` - Contains `FrameEncoder`, `TemporalPredictor`, and `JEPAVideoModel`
- `lightning_module.py` - PyTorch Lightning wrapper for training

### Model Components:
- **FrameEncoder**: Wraps any image encoder and projects to embedding dimension
- **TemporalPredictor**: Transformer-based predictor with learnable temporal position embeddings
- **JEPAVideoModel**: Complete JEPA model with student-teacher architecture and EMA updates

# Training
At the moment, the implemented losses are limited to JEPA-like training.

**Training script**: `train_video_embeddings.py`

### JEPA-like embedding training 
JEPA-like training with a frame encoder teacher and student network and a transformer predictor with time-embeddings.
This loss trains the student to predict the teacher embeddings of future frames given past frames. 
It does not have local (dense) losses.

### Training Features:
- Multi-GPU support via PyTorch Lightning
- Mixed precision training (fp16, bf16)
- Gradient accumulation
- Learning rate warmup with cosine annealing
- EMA updates for teacher encoder
- Optional GPU augmentation using Kornia
- TensorBoard logging
- Optional ClearML integration

### Example Usage:

```bash
# Basic training
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --val-data-path /path/to/val.lmdb \
    --name video_jepa_resnet34 \
    --encoder-config '{"type":"sm","name":"resnet34","weights":"imagenet","depth":5}' \
    --embedding-dim 256 \
    --num-frames 8 \
    --context-frames 4 \
    --batch-size 16 \
    --gpus 1 \
    --precision 16-mixed

# Multi-GPU training
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --name video_jepa_multigpu \
    --gpus 4 \
    --batch-size 8 \
    --accumulate-grad-batches 2

# With GPU augmentation
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --gpu-augmentation aug_1 \
    --gpus 1
```

# Augmentation
A GPU based augmentation pipeline is implemented in `common/augmentations_gpu.py` using Kornia.
Can be enabled with `--gpu-augmentation aug_1` flag.

# Inference and Export

**Utilities**: `utils.py` - Helper functions for model export and inference

### Export trained encoder:
```python
from utils import export_encoder

export_encoder(
    checkpoint_path='lightning_logs/model/checkpoints/last.ckpt',
    output_path='encoder.pt',
    trace=True
)
```

### Load for inference:
```python
from utils import load_encoder_for_inference, extract_video_embedding

encoder = load_encoder_for_inference('encoder.pt', device='cuda')
embedding = extract_video_embedding(encoder, frames, aggregate='mean')
```

### Video-level encoder wrapper:
```python
from utils import VideoEncoder

video_encoder = VideoEncoder(frame_encoder, aggregate='mean', normalize=True)
video_embeddings = video_encoder(batch_of_videos)  # (B, T, C, H, W) -> (B, D)
```


