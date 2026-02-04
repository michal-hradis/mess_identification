"""PyTorch Lightning module for JEPA video embedding training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Dict, Any


class JEPALoss(nn.Module):
    """JEPA prediction loss (smooth L1 or L2)."""

    def __init__(self, loss_type: str = "smooth_l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, T, D) predicted embeddings
            targets: (B, T, D) target embeddings

        Returns:
            loss: scalar loss value
        """
        if self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(predictions, targets)
        elif self.loss_type == "mse":
            loss = F.mse_loss(predictions, targets)
        elif self.loss_type == "cosine":
            # Negative cosine similarity
            predictions_norm = F.normalize(predictions, p=2, dim=-1)
            targets_norm = F.normalize(targets, p=2, dim=-1)
            loss = 1 - (predictions_norm * targets_norm).sum(dim=-1).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class VideoEmbeddingModule(pl.LightningModule):
    """PyTorch Lightning module for video embedding training."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        context_frames: int = 4,
        loss_type: str = "smooth_l1",
        ema_decay: float = 0.999,
        ema_update_every: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.loss_fn = JEPALoss(loss_type=loss_type)
        self.context_frames = context_frames
        self.ema_update_every = ema_update_every

        # GPU augmentation (optional)
        self.augmentation = None

    def set_augmentation(self, augmentation: Optional[nn.Module]):
        """Set GPU-based augmentation pipeline."""
        self.augmentation = augmentation
        if augmentation is not None:
            self.augmentation = self.augmentation.to(self.device)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(frames, context_frames=self.context_frames)

    def apply_augmentation(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply GPU augmentation if available."""
        if self.augmentation is None:
            return frames

        B, T, C, H, W = frames.shape

        # Convert to float [0, 1] for augmentation
        frames_float = frames.float() / 255.0

        # Flatten batch and time for augmentation
        frames_flat = frames_float.view(B * T, C, H, W)

        # Apply augmentation
        frames_aug = self.augmentation(frames_flat)

        # Reshape back and convert to uint8
        frames_aug = frames_aug.view(B, T, C, H, W)
        frames_aug = (frames_aug * 255).clamp(0, 255).to(torch.uint8)

        return frames_aug

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        frames = batch['frames']  # (B, T, C, H, W)

        # Apply augmentation
        frames = self.apply_augmentation(frames)

        # Forward pass
        predictions, targets = self(frames)

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        # Log metrics
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        # Compute additional metrics
        with torch.no_grad():
            # Cosine similarity between predictions and targets
            pred_norm = F.normalize(predictions, p=2, dim=-1)
            tgt_norm = F.normalize(targets, p=2, dim=-1)
            cosine_sim = (pred_norm * tgt_norm).sum(dim=-1).mean()
            self.log('train/cosine_sim', cosine_sim, on_step=True, on_epoch=True)

            # L2 distance
            l2_dist = torch.norm(predictions - targets, p=2, dim=-1).mean()
            self.log('train/l2_distance', l2_dist, on_step=True, on_epoch=True)

        # Update teacher with EMA
        if (self.global_step + 1) % self.ema_update_every == 0:
            self.model.update_teacher()

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        frames = batch['frames']

        # Forward pass (no augmentation for validation)
        predictions, targets = self(frames)

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        # Log metrics
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        # Compute additional metrics
        with torch.no_grad():
            pred_norm = F.normalize(predictions, p=2, dim=-1)
            tgt_norm = F.normalize(targets, p=2, dim=-1)
            cosine_sim = (pred_norm * tgt_norm).sum(dim=-1).mean()
            self.log('val/cosine_sim', cosine_sim, on_step=False, on_epoch=True)

            l2_dist = torch.norm(predictions - targets, p=2, dim=-1).mean()
            self.log('val/l2_distance', l2_dist, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters: predictor and student encoder
        params = [
            {'params': self.model.student_encoder.parameters()},
            {'params': self.model.predictor.parameters()}
        ]

        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / self.hparams.warmup_steps
            else:
                # Cosine decay after warmup
                progress = (step - self.hparams.warmup_steps) / max(1, self.trainer.max_steps - self.hparams.warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log learning rate."""
        lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', lr, on_step=True, on_epoch=False)
