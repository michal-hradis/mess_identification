"""Quick reference and common commands for video embeddings."""

# QUICK REFERENCE - VIDEO EMBEDDINGS

## Training Commands

# Basic training (ResNet34, 8 frames, 1 GPU)
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --name my_model \
    --embedding-dim 256 \
    --batch-size 16 \
    --gpus 1

# With validation
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --val-data-path /path/to/val.lmdb \
    --name my_model \
    --val-check-interval 1000 \
    --gpus 1

# Multi-GPU (4 GPUs)
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --name my_model_4gpu \
    --gpus 4 \
    --batch-size 8 \
    --accumulate-grad-batches 2

# With GPU augmentation
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --name my_model_aug \
    --gpu-augmentation aug_1 \
    --gpus 1

# Different encoder (EfficientNet)
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --name efficientnet_model \
    --encoder-config '{"type":"sm","name":"efficientnet-b0","weights":"imagenet","depth":5}' \
    --embedding-dim 256 \
    --gpus 1

# Larger model (ResNet50, 512 dims, more frames)
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --name large_model \
    --encoder-config '{"type":"sm","name":"resnet50","weights":"imagenet","depth":5}' \
    --embedding-dim 512 \
    --num-frames 16 \
    --context-frames 6 \
    --batch-size 8 \
    --gpus 2

# With ClearML tracking
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --name tracked_model \
    --use-clearml \
    --project "VideoEmbeddings" \
    --gpus 1

# Resume training
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --name my_model \
    --resume-from lightning_logs/my_model/checkpoints/last.ckpt \
    --gpus 1

# Long training with checkpoints
python train_video_embeddings.py \
    --data-path /path/to/train.lmdb \
    --name long_training \
    --max-steps 500000 \
    --checkpoint-every 5000 \
    --val-check-interval 2000 \
    --gpus 1

## Export Commands

# Export trained model
python utils.py \
    --checkpoint lightning_logs/my_model/checkpoints/last.ckpt \
    --output my_encoder.pt

# Export specific checkpoint
python utils.py \
    --checkpoint lightning_logs/my_model/checkpoints/jepa-0050000.ckpt \
    --output encoder_50k.pt

# Export without tracing (state dict only)
python utils.py \
    --checkpoint lightning_logs/my_model/checkpoints/last.ckpt \
    --output encoder_state.pt \
    --no-trace

## Inference Code Snippets

### Load and use encoder
```python
from video_embeddings import load_encoder_for_inference, extract_video_embedding
import torch

# Load encoder
encoder = load_encoder_for_inference('my_encoder.pt', device='cuda')

# Process single video
frames = torch.randint(0, 256, (16, 3, 224, 224), dtype=torch.uint8).cuda()
embedding = extract_video_embedding(encoder, frames, aggregate='mean')
print(embedding.shape)  # (256,)
```

### Batch processing
```python
from video_embeddings import VideoEncoder
import torch

encoder = load_encoder_for_inference('my_encoder.pt', device='cuda')
video_encoder = VideoEncoder(encoder, aggregate='mean', normalize=True)

# Process batch
videos = torch.randint(0, 256, (8, 16, 3, 224, 224), dtype=torch.uint8).cuda()
embeddings = video_encoder(videos)
print(embeddings.shape)  # (8, 256)
```

### Load dataset
```python
from video_embeddings import VideoDataset
from torch.utils.data import DataLoader

dataset = VideoDataset(
    data_path='/path/to/data.lmdb',
    num_frames=8,
    max_frame_gap=5,
    image_size=(224, 224),
    is_lmdb=True
)

loader = DataLoader(dataset, batch_size=4, num_workers=2)

for batch in loader:
    frames = batch['frames']  # (4, 8, 3, 224, 224)
    video_ids = batch['video_id']  # (4,)
    break
```

## Monitoring

# TensorBoard
tensorboard --logdir lightning_logs

# Watch specific run
tensorboard --logdir lightning_logs/my_model

## Testing

# Run all tests
cd video_embeddings
python test_components.py

# Test import
python -c "from video_embeddings import *; print('✓ OK')"

## Common Configurations

# Small/Fast (for testing)
--encoder-config '{"type":"sm","name":"resnet18","weights":"imagenet","depth":5}'
--embedding-dim 128
--num-frames 4
--batch-size 32

# Medium (recommended)
--encoder-config '{"type":"sm","name":"resnet34","weights":"imagenet","depth":5}'
--embedding-dim 256
--num-frames 8
--batch-size 16

# Large (best quality)
--encoder-config '{"type":"sm","name":"resnet50","weights":"imagenet","depth":5}'
--embedding-dim 512
--num-frames 16
--batch-size 8

# EfficientNet (efficient)
--encoder-config '{"type":"sm","name":"efficientnet-b0","weights":"imagenet","depth":5}'
--embedding-dim 256
--num-frames 8
--batch-size 24

## Troubleshooting

# Out of memory? Reduce batch size or use gradient accumulation
--batch-size 4 --accumulate-grad-batches 4

# Slow training? Use mixed precision
--precision 16-mixed

# Not enough data? Use augmentation
--gpu-augmentation aug_1

# Want better quality? Increase model size
--encoder-config '{"type":"sm","name":"resnet50",...}'
--embedding-dim 512
--num-frames 16

## File Locations

# Checkpoints
lightning_logs/{name}/checkpoints/

# TensorBoard logs
lightning_logs/{name}/version_*/

# Last checkpoint
lightning_logs/{name}/checkpoints/last.ckpt

# Specific step checkpoint
lightning_logs/{name}/checkpoints/jepa-{step:07d}.ckpt
