#!/bin/bash
# Example training script with Domain Adaptation enabled

# Basic example with domain adaptation
python train_id.py \
  --lmdb /path/to/train.lmdb \
  --lmdb-tst /path/to/test.lmdb \
  --use-domain-adaptation \
  --domain-loss-weight 0.1 \
  --backbone-config '{"type":"SM","name":"resnet34","weights":"imagenet","depth":5}' \
  --decoder-config '{"type":"pool","operation":"avg"}' \
  --emb-dim 256 \
  --batch-size 128 \
  --learning-rate 0.0001 \
  --name my_robust_model

# Advanced example with custom domain classifier
python train_id.py \
  --lmdb /path/to/train.lmdb \
  --lmdb-tst /path/to/test.lmdb \
  --use-domain-adaptation \
  --domain-loss-weight 0.2 \
  --domain-hidden-dims 512 256 128 \
  --domain-dropout 0.3 \
  --domain-grl-lambda 1.5 \
  --backbone-config '{"type":"SM","name":"resnet50","weights":"imagenet","depth":5}' \
  --decoder-config '{"type":"pool","operation":"avg"}' \
  --emb-dim 512 \
  --batch-size 96 \
  --learning-rate 0.0001 \
  --augmentation aug_hard \
  --loss arcface \
  --name my_advanced_robust_model

# Training without domain adaptation (original behavior)
python train_id.py \
  --lmdb /path/to/train.lmdb \
  --lmdb-tst /path/to/test.lmdb \
  --backbone-config '{"type":"SM","name":"resnet34","weights":"imagenet","depth":5}' \
  --decoder-config '{"type":"pool","operation":"avg"}' \
  --emb-dim 256 \
  --batch-size 128 \
  --learning-rate 0.0001 \
  --name my_standard_model

