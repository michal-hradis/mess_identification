"""Training script for video embedding learning with JEPA."""
import argparse
import logging
from pathlib import Path
import json

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from video_dataset import VideoDataset
from jepa_model import FrameEncoder, TemporalPredictor, JEPAVideoModel
from lightning_module import VideoEmbeddingModule


def create_encoder_from_config(config: dict, embedding_dim: int):
    """Create encoder from configuration."""
    encoder_type = config.get('type', 'pretrained').lower()

    if encoder_type == 'pretrained' or encoder_type == 'sm':
        # Use segmentation_models_pytorch encoder
        from common.nets_pretrained import PretrainedEncoder

        name = config.get('name', 'resnet34')
        depth = config.get('depth', 5)
        weights = config.get('weights', 'imagenet')

        base_encoder = PretrainedEncoder(
            name=name,
            depth=depth,
            weights=weights,
            in_channels=3
        )

    elif encoder_type == 'timm':
        # Use timm model
        import timm

        model_name = config.get('name', 'resnet34')
        pretrained = config.get('pretrained', True)

        base_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''   # Remove global pooling
        )

    elif encoder_type == 'torchvision':
        # Use torchvision model
        import torchvision.models as models

        model_name = config.get('name', 'resnet34')
        weights = config.get('weights', 'DEFAULT')

        # Get model class
        model_class = getattr(models, model_name)

        if weights == 'DEFAULT':
            weights = getattr(models, f"{model_name.upper()}_Weights").DEFAULT
        elif weights is None or weights == 'none':
            weights = None

        base_model = model_class(weights=weights)

        # Remove final classification layer
        if hasattr(base_model, 'fc'):
            base_encoder = torch.nn.Sequential(*list(base_model.children())[:-1])
        elif hasattr(base_model, 'classifier'):
            base_encoder = torch.nn.Sequential(*list(base_model.children())[:-1])
        else:
            base_encoder = base_model

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return base_encoder


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train video embeddings with JEPA-like approach'
    )

    # Data arguments
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to LMDB or directory with video frames')
    parser.add_argument('--val-data-path', type=str, default=None,
                        help='Path to validation data (optional)')
    parser.add_argument('--is-lmdb', action='store_true', default=True,
                        help='Whether data path points to LMDB')
    parser.add_argument('--no-lmdb', dest='is_lmdb', action='store_false',
                        help='Data path is a directory, not LMDB')

    # Dataset arguments
    parser.add_argument('--num-frames', type=int, default=8,
                        help='Number of frames to sample per video')
    parser.add_argument('--max-frame-gap', type=int, default=5,
                        help='Maximum gap between consecutive frames')
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224],
                        help='Image size (height width)')

    # Model arguments
    parser.add_argument('--encoder-config', type=str,
                        default='{"type":"sm","name":"resnet34","weights":"imagenet","depth":5}',
                        help='Encoder configuration JSON')
    parser.add_argument('--embedding-dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--predictor-heads', type=int, default=8,
                        help='Number of attention heads in predictor')
    parser.add_argument('--predictor-layers', type=int, default=4,
                        help='Number of transformer layers in predictor')
    parser.add_argument('--predictor-dropout', type=float, default=0.1,
                        help='Dropout in predictor')
    parser.add_argument('--context-frames', type=int, default=4,
                        help='Number of context frames for prediction')
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay rate for teacher encoder')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--max-steps', type=int, default=100000,
                        help='Maximum number of training steps')
    parser.add_argument('--loss-type', type=str, default='smooth_l1',
                        choices=['smooth_l1', 'mse', 'cosine'],
                        help='Loss function type')

    # GPU augmentation
    parser.add_argument('--gpu-augmentation', type=str, default=None,
                        help='GPU augmentation pipeline name')

    # Logging and checkpointing
    parser.add_argument('--name', type=str, default='video_jepa',
                        help='Experiment name')
    parser.add_argument('--save-dir', type=str, default='./lightning_logs',
                        help='Directory to save logs and checkpoints')
    parser.add_argument('--val-check-interval', type=int, default=1000,
                        help='Validation check interval (steps)')
    parser.add_argument('--checkpoint-every', type=int, default=5000,
                        help='Save checkpoint every N steps')

    # PyTorch Lightning arguments
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        choices=['32', '16-mixed', 'bf16-mixed'],
                        help='Training precision')
    parser.add_argument('--accumulate-grad-batches', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # ClearML logging
    parser.add_argument('--project', type=str, default=None,
                        help='ClearML project name')
    parser.add_argument('--use-clearml', action='store_true',
                        help='Enable ClearML logging')

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Parse encoder config
    encoder_config = json.loads(args.encoder_config)
    logging.info(f"Encoder config: {encoder_config}")

    # Initialize ClearML if requested
    clearml_logger = None
    if args.use_clearml and args.project:
        try:
            from clearml import Task
            task = Task.init(project_name=args.project, task_name=args.name)
            task.connect(args)
            logging.info(f"ClearML initialized: project={args.project}, task={args.name}")
        except ImportError:
            logging.warning("ClearML not installed, skipping ClearML logging")

    # Create datasets
    train_dataset = VideoDataset(
        data_path=args.data_path,
        num_frames=args.num_frames,
        max_frame_gap=args.max_frame_gap,
        image_size=tuple(args.image_size),
        is_lmdb=args.is_lmdb
    )
    logging.info(f"Train dataset: {len(train_dataset)} videos")

    val_dataset = None
    if args.val_data_path:
        val_dataset = VideoDataset(
            data_path=args.val_data_path,
            num_frames=args.num_frames,
            max_frame_gap=args.max_frame_gap,
            image_size=tuple(args.image_size),
            is_lmdb=args.is_lmdb
        )
        logging.info(f"Val dataset: {len(val_dataset)} videos")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0
        )

    # Create model components
    base_encoder = create_encoder_from_config(encoder_config, args.embedding_dim)

    student_encoder = FrameEncoder(
        base_encoder=base_encoder,
        embedding_dim=args.embedding_dim,
        normalize=True
    )

    # Create teacher encoder (copy of student)
    teacher_base_encoder = create_encoder_from_config(encoder_config, args.embedding_dim)
    teacher_encoder = FrameEncoder(
        base_encoder=teacher_base_encoder,
        embedding_dim=args.embedding_dim,
        normalize=True
    )

    predictor = TemporalPredictor(
        embedding_dim=args.embedding_dim,
        num_heads=args.predictor_heads,
        num_layers=args.predictor_layers,
        dropout=args.predictor_dropout,
        max_frames=args.num_frames
    )

    model = JEPAVideoModel(
        student_encoder=student_encoder,
        teacher_encoder=teacher_encoder,
        predictor=predictor,
        ema_decay=args.ema_decay
    )

    # Create Lightning module
    pl_module = VideoEmbeddingModule(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        context_frames=args.context_frames,
        loss_type=args.loss_type,
        ema_decay=args.ema_decay
    )

    # Set GPU augmentation if specified
    if args.gpu_augmentation:
        from common.augmentations_gpu import GPU_AUGMENTATIONS
        augmentation = GPU_AUGMENTATIONS[args.gpu_augmentation]
        pl_module.set_augmentation(augmentation)
        logging.info(f"Using GPU augmentation: {args.gpu_augmentation}")

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(args.save_dir) / args.name / 'checkpoints',
            filename='jepa-{step:07d}',
            save_top_k=-1,
            every_n_train_steps=args.checkpoint_every,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # Setup logger
    loggers = [
        TensorBoardLogger(
            save_dir=args.save_dir,
            name=args.name
        )
    ]

    # Create trainer
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 'auto',
        precision=args.precision,
        callbacks=callbacks,
        logger=loggers,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        enable_model_summary=True
    )

    # Train
    logging.info("Starting training...")
    trainer.fit(
        pl_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from
    )

    logging.info("Training completed!")


if __name__ == '__main__':
    main()
