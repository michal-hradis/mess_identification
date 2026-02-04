# Getting Started Checklist - Video Embeddings

## ✅ Pre-flight Checklist

### 1. Installation
- [ ] Python 3.8+ installed
- [ ] PyTorch installed (`pip install torch torchvision`)
- [ ] PyTorch Lightning installed (`pip install pytorch-lightning`)
- [ ] LMDB installed (`pip install lmdb`)
- [ ] OpenCV installed (`pip install opencv-python`)
- [ ] Segmentation Models PyTorch installed (`pip install segmentation-models-pytorch`)
- [ ] (Optional) Kornia installed for GPU augmentation (`pip install kornia`)
- [ ] (Optional) ClearML installed for tracking (`pip install clearml`)
- [ ] (Optional) TIMM installed for more encoders (`pip install timm`)

### 2. Data Preparation
- [ ] Video frames extracted to images
- [ ] Files named as `{video_id}_{frame_id}.jpg`
- [ ] Data organized in LMDB or directory
- [ ] At least 8 frames per video (or adjust `--num-frames`)
- [ ] Images are readable and not corrupted
- [ ] (Optional) Separate validation dataset prepared

### 3. Verify Installation
```bash
cd video_embeddings
python -c "from video_embeddings import *; print('✓ Installation OK')"
```

### 4. Run Tests (Optional but Recommended)
```bash
cd video_embeddings
python test_components.py
```

## 🚀 Quick Start (5 minutes)

### Step 1: Basic Training
```bash
python train_video_embeddings.py \
    --data-path /path/to/your/data.lmdb \
    --name my_first_model \
    --embedding-dim 256 \
    --num-frames 8 \
    --batch-size 16 \
    --max-steps 1000 \
    --gpus 1
```

### Step 2: Monitor Training
```bash
# In another terminal
tensorboard --logdir lightning_logs
# Open http://localhost:6006 in browser
```

### Step 3: Export Model
```bash
python utils.py \
    --checkpoint lightning_logs/my_first_model/checkpoints/last.ckpt \
    --output my_encoder.pt
```

### Step 4: Test Inference
```python
from video_embeddings import load_encoder_for_inference, extract_video_embedding
import torch

encoder = load_encoder_for_inference('my_encoder.pt', device='cuda')
frames = torch.randint(0, 256, (8, 3, 224, 224), dtype=torch.uint8).cuda()
embedding = extract_video_embedding(encoder, frames, aggregate='mean')
print(f"Embedding shape: {embedding.shape}")  # Should be (256,)
print("✓ Inference working!")
```

## 📋 Common Issues and Solutions

### Issue: Out of Memory
**Solution**: Reduce batch size or use gradient accumulation
```bash
--batch-size 8 --accumulate-grad-batches 2
```

### Issue: Training Too Slow
**Solution**: Use mixed precision
```bash
--precision 16-mixed
```

### Issue: "No module named 'video_embeddings'"
**Solution**: Make sure you're in the correct directory
```bash
cd /path/to/mess_identification
python video_embeddings/train_video_embeddings.py ...
```

### Issue: LMDB errors
**Solution**: Check LMDB path and permissions
```python
import lmdb
env = lmdb.open('/path/to/data.lmdb', readonly=True)
print(env.stat())
env.close()
```

### Issue: Not enough frames in videos
**Solution**: Reduce `--num-frames` parameter
```bash
--num-frames 4
```

### Issue: Validation errors
**Solution**: Make sure validation data path is correct
```bash
--val-data-path /path/to/val.lmdb
```

## 🎯 Next Steps

### For Experimentation
1. [ ] Try different encoder architectures
2. [ ] Experiment with different embedding dimensions
3. [ ] Test different loss functions (smooth_l1, mse, cosine)
4. [ ] Try GPU augmentation (`--gpu-augmentation aug_1`)
5. [ ] Adjust learning rate and warmup steps

### For Production
1. [ ] Set up ClearML for experiment tracking
2. [ ] Use multi-GPU training for faster iterations
3. [ ] Set up proper validation dataset
4. [ ] Monitor validation metrics
5. [ ] Export best checkpoint (not just last)
6. [ ] Document your best configuration

### For Research
1. [ ] Implement custom augmentations
2. [ ] Try different temporal predictors
3. [ ] Experiment with masking strategies
4. [ ] Add downstream evaluation tasks
5. [ ] Compare with other methods

## 📖 Documentation Reference

- **README.md** - Complete user guide
- **IMPLEMENTATION_SUMMARY.md** - Technical overview
- **QUICK_REFERENCE.py** - Common commands
- **video_embeddings.md** - Original specification
- **test_components.py** - Usage examples

## 🆘 Getting Help

### Check Logs
```bash
# TensorBoard
tensorboard --logdir lightning_logs

# Console output
# Training script prints progress and metrics
```

### Debug Mode
```python
# Test dataset loading
from video_embeddings import VideoDataset
ds = VideoDataset('/path/to/data.lmdb', num_frames=8)
print(f"Dataset size: {len(ds)}")
batch = ds[0]
print(f"Batch keys: {batch.keys()}")
print(f"Frames shape: {batch['frames'].shape}")
```

### Validate Configuration
```bash
# Check encoder config
python -c "import json; print(json.loads('{\"type\":\"sm\",\"name\":\"resnet34\"}'))"

# Check GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## ✨ Tips for Success

1. **Start small**: Test with small model and few steps first
2. **Monitor metrics**: Watch training loss and cosine similarity
3. **Use validation**: Always validate on held-out data
4. **Save checkpoints**: Don't lose your progress
5. **Document experiments**: Use ClearML or keep notes
6. **GPU augmentation**: Much faster than CPU augmentation
7. **Mixed precision**: 2-3x speedup on modern GPUs
8. **Persistent workers**: Already enabled in dataset
9. **Pin memory**: Already enabled in data loaders
10. **Batch size**: Start with 16, adjust based on GPU memory

## 🎓 Learning Resources

### Understanding JEPA
- Student-teacher architecture learns by predicting
- EMA updates keep teacher stable
- Temporal predictor models video dynamics
- No contrastive negatives needed

### PyTorch Lightning
- Automatic multi-GPU scaling
- Built-in logging and checkpointing
- Easy to customize and extend

### Key Metrics
- **Loss**: Should decrease steadily
- **Cosine similarity**: Between predictions and targets
- **L2 distance**: Between predictions and targets
- **Learning rate**: Should increase during warmup

## ✅ Success Criteria

Training is working well when:
- [ ] Loss decreases over time
- [ ] Cosine similarity increases (approaches 1.0)
- [ ] L2 distance decreases
- [ ] No NaN or Inf values
- [ ] GPU utilization is high (>80%)
- [ ] Validation metrics are reasonable

## 🎉 You're Ready!

With this checklist complete, you should be able to:
- ✅ Train video embedding models
- ✅ Export and use trained models
- ✅ Monitor and debug training
- ✅ Customize for your needs

**Happy training! 🚀**
