# Domain Adaptation with Gradient Reversal Layer

This document describes the Domain Adaptation feature added to improve model robustness by reducing domain-specific information in learned embeddings.

## Overview

The Domain Adaptation feature uses a **Gradient Reversal Layer (GRL)** to encourage the model to learn embeddings that contain semantic information while being invariant to the source video/domain. This is achieved through adversarial training where:

1. A domain classifier (MLP head) tries to predict the video_id from embeddings
2. The gradient reversal layer inverts gradients during backpropagation
3. The main model learns to produce embeddings that fool the domain classifier
4. Result: Embeddings contain less domain-specific information

## Architecture

### Gradient Reversal Layer
- Acts as identity during forward pass
- Multiplies gradients by -λ during backward pass
- Encourages domain-invariant feature learning

### Domain Classifier
- Multi-layer perceptron (MLP) with configurable architecture
- Input: Image embeddings
- Output: Video ID classification logits
- Includes batch normalization, ReLU activations, and dropout

### Model Wrapper
- `EmbeddingModelWithDomainAdaptation` wraps the base embedding model
- Optionally adds domain classifier head
- Base model can be extracted for export (without domain head)

## Usage

### Command-Line Arguments

Enable domain adaptation with these arguments:

```bash
python train_id.py \
  --lmdb /path/to/train.lmdb \
  --use-domain-adaptation \
  --domain-loss-weight 0.1 \
  --domain-hidden-dims 256 128 \
  --domain-dropout 0.5 \
  --domain-grl-lambda 1.0 \
  [other arguments...]
```

### Arguments Description

- `--use-domain-adaptation`: Enable domain adaptation (flag, default: False)
- `--domain-loss-weight`: Weight for domain adaptation loss (default: 0.1)
  - Higher values = stronger domain invariance
  - Typical range: 0.01 - 1.0
- `--domain-hidden-dims`: Hidden layer sizes for domain classifier MLP (default: [256, 128])
  - Example: `--domain-hidden-dims 512 256 128`
- `--domain-dropout`: Dropout probability in domain classifier (default: 0.5)
- `--domain-grl-lambda`: Gradient reversal strength (default: 1.0)
  - Higher values = stronger gradient reversal
  - Can be adjusted during training for curriculum learning

## Training Details

### Loss Function

Total training loss:
```
total_loss = embedding_loss + domain_loss_weight * domain_loss + emb_reg_weight * regularization
```

Where:
- `embedding_loss`: Main contrastive/metric learning loss (xent, arcface, etc.)
- `domain_loss`: Cross-entropy loss for video_id classification
- Gradients from `domain_loss` are reversed before reaching the encoder

### Training Flow

1. Forward pass: Get embeddings and domain predictions
2. Compute embedding loss (identity classification)
3. Compute domain loss (video_id classification)
4. Backward pass: 
   - Domain classifier learns to classify videos
   - Encoder learns to confuse domain classifier (via GRL)
5. Result: Embeddings are good for identity but poor for video classification

## Model Export

The exported models **do not include the domain adaptation head**:

- Checkpoint files (`.ckpt`) contain only the base embedding model
- Traced models (`.pt`) contain only the base embedding model
- This ensures deployment models are lightweight and focused

The domain adaptation head is used only during training to improve robustness.

## Expected Benefits

1. **Cross-domain generalization**: Better performance on new video sources
2. **Reduced overfitting**: Less memorization of domain-specific artifacts
3. **Robust embeddings**: Focus on semantic content rather than recording conditions
4. **Fair comparisons**: Less bias toward specific capture devices or conditions

## Hyperparameter Tuning

### Starting Point
```bash
--use-domain-adaptation \
--domain-loss-weight 0.1 \
--domain-hidden-dims 256 128 \
--domain-dropout 0.5
```

### If domain classifier is too weak (domain loss high):
- Increase `--domain-hidden-dims` (e.g., 512 256 128)
- Decrease `--domain-dropout`
- Increase `--domain-loss-weight`

### If embeddings lose semantic information (poor retrieval):
- Decrease `--domain-loss-weight`
- Reduce `--domain-grl-lambda`
- Simplify domain classifier (fewer/smaller hidden layers)

## Implementation Files

- `domain_adaptation.py`: Contains GRL and domain classifier implementations
- `train_id.py`: Updated training script with domain adaptation integration

## Reference

Based on: "Unsupervised Domain Adaptation by Backpropagation" (Ganin & Lempitsky, 2015)

