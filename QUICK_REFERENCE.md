# Domain Adaptation Quick Reference

## TL;DR - Just Tell Me How To Use It!

### Enable Domain Adaptation (Simplest)
```bash
python train_id.py --lmdb /path/to/data.lmdb --use-domain-adaptation
```

### With Custom Settings
```bash
python train_id.py \
  --lmdb /path/to/data.lmdb \
  --use-domain-adaptation \
  --domain-loss-weight 0.1 \
  --domain-hidden-dims 256 128
```

## What Does It Do?

Makes your embeddings **ignore the video source** (camera, lighting, compression) and focus only on **semantic content** (person/object identity).

## When Should I Use It?

✅ Training data from multiple video sources  
✅ Want better cross-domain generalization  
✅ Testing on different cameras than training  
✅ Want more robust embeddings  

❌ Single video source  
❌ Domain information is useful  
❌ Very small dataset  

## Key Parameters

| Parameter | What It Does | Values |
|-----------|--------------|--------|
| `--domain-loss-weight` | How much to ignore domains | 0.05-0.2 (start 0.1) |
| `--domain-hidden-dims` | Classifier complexity | 256 128 (default) |
| `--domain-dropout` | Regularization | 0.3-0.5 (default 0.5) |
| `--domain-grl-lambda` | Reversal strength | 1.0 (default) |

## Quick Troubleshooting

**Performance dropped?**  
→ Lower `--domain-loss-weight` to 0.05

**Domain loss not decreasing?**  
→ Increase `--domain-loss-weight` to 0.15

**Training unstable?**  
→ Reduce learning rate

## What Gets Exported?

✅ Clean base model (NO domain head)  
✅ Same size as without domain adaptation  
✅ Drop-in replacement for inference  

## Example Commands

### Basic
```bash
python train_id.py --lmdb train.lmdb --use-domain-adaptation --name my_model
```

### With Testing
```bash
python train_id.py \
  --lmdb train.lmdb \
  --lmdb-tst test.lmdb \
  --use-domain-adaptation \
  --domain-loss-weight 0.1 \
  --name my_robust_model
```

### Advanced
```bash
python train_id.py \
  --lmdb train.lmdb \
  --use-domain-adaptation \
  --domain-loss-weight 0.15 \
  --domain-hidden-dims 512 256 128 \
  --domain-dropout 0.3 \
  --emb-dim 512 \
  --batch-size 96 \
  --name my_advanced_model
```

### Original (No Domain Adaptation)
```bash
python train_id.py --lmdb train.lmdb --name baseline_model
```

## Files You Need

**Must have:**
- `domain_adaptation.py` (in same directory)
- Updated `train_id.py`

**Documentation:**
- `DOMAIN_ADAPTATION_README.md` (detailed docs)
- `IMPLEMENTATION_SUMMARY.md` (technical details)

## How To Check It's Working

Look for these log messages:
```
INFO:root:Domain Adaptation enabled with X domains (video_ids)
INFO:root:Domain classifier hidden dims: [256, 128]
INFO:root:Domain loss weight: 0.1
```

During training, both losses should decrease.

## Performance Tips

1. **Start with defaults** - Works well for most cases
2. **Lower weight first** - If performance drops, reduce domain loss weight
3. **Monitor both losses** - Main loss and domain loss should both decrease
4. **Compare results** - Train with/without DA to see improvement
5. **Test cross-domain** - That's where you'll see the biggest gains

## Common Patterns

### Light Domain Adaptation
```bash
--use-domain-adaptation --domain-loss-weight 0.05
```

### Standard Domain Adaptation (Recommended)
```bash
--use-domain-adaptation --domain-loss-weight 0.1
```

### Strong Domain Adaptation
```bash
--use-domain-adaptation --domain-loss-weight 0.2 --domain-hidden-dims 512 256
```

---

**That's it!** Add `--use-domain-adaptation` and you're good to go. 🚀

For more details, see `DOMAIN_ADAPTATION_README.md`

