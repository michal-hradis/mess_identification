"""JEPA-like video embedding model components.

This module implements a student-teacher architecture for video embedding learning:
- FrameEncoder: Wraps any image encoder and projects to embedding dimension
- TemporalPredictor: Transformer-based predictor with learnable temporal position embeddings
- JEPAVideoModel: Complete model with EMA-updated teacher encoder
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FrameEncoder(nn.Module):
    """Frame encoder that wraps any image encoder and projects to embedding dimension.

    Args:
        base_encoder: Any image encoder (e.g., from segmentation_models_pytorch, timm, torchvision)
        embedding_dim: Target embedding dimension
        normalize: Whether to L2-normalize embeddings
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        embedding_dim: int = 256,
        normalize: bool = True
    ):
        super().__init__()
        self.base_encoder = base_encoder
        self.embedding_dim = embedding_dim
        self.normalize = normalize

        # Determine output dimension of base encoder
        # We'll do a test forward pass with a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            try:
                features = base_encoder(dummy_input)
                # Handle different encoder output formats
                if isinstance(features, (list, tuple)):
                    features = features[-1]  # Take last feature map
                if len(features.shape) == 4:  # (B, C, H, W)
                    encoder_dim = features.shape[1]
                    self.pool = nn.AdaptiveAvgPool2d((1, 1))
                elif len(features.shape) == 2:  # (B, C)
                    encoder_dim = features.shape[1]
                    self.pool = None
                else:
                    raise ValueError(f"Unexpected encoder output shape: {features.shape}")
            except Exception as e:
                logging.error(f"Failed to determine encoder output dimension")
                raise ValueError(f"Error determining encoder output dimension: {e}") from e

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # Initialize projection with small weights
        with torch.no_grad():
            self.projection[0].weight.data *= 0.1
            if self.projection[0].bias is not None:
                self.projection[0].bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images, uint8 tensor of shape (B, 3, H, W) or float tensor in [0, 1]

        Returns:
            embeddings: Tensor of shape (B, embedding_dim)
        """
        # Convert uint8 to float [0, 1] if needed
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Extract features
        features = self.base_encoder(x)

        # Handle different encoder output formats
        if isinstance(features, (list, tuple)):
            features = features[-1]

        # Pool if needed
        if self.pool is not None:
            features = self.pool(features)

        # Flatten
        features = features.view(features.size(0), -1)

        # Project to embedding dimension
        embeddings = self.projection(features)

        # Normalize if requested
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


class TemporalPredictor(nn.Module):
    """Transformer-based temporal predictor with learnable position embeddings.

    Predicts future frame embeddings from past frame embeddings using a transformer architecture.

    Args:
        embedding_dim: Dimension of frame embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        max_frames: Maximum number of frames (for positional embeddings)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_frames: int = 32
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_frames = max_frames

        # Learnable temporal position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(max_frames, embedding_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        context_embeddings: torch.Tensor,
        target_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            context_embeddings: Context frame embeddings, shape (B, T_context, D)
            target_positions: Target frame positions (indices), shape (B, T_target)

        Returns:
            predictions: Predicted embeddings for target frames, shape (B, T_target, D)
        """
        B, T_context, D = context_embeddings.shape

        # Add positional embeddings to context
        context_pos = torch.arange(T_context, device=context_embeddings.device)
        context_with_pos = context_embeddings + self.pos_embedding[context_pos].unsqueeze(0)

        # Process through transformer
        context_features = self.transformer(context_with_pos)

        # Create queries for target positions
        target_pos_emb = self.pos_embedding[target_positions]  # (B, T_target, D)

        # Use mean of context features as base for prediction
        context_mean = context_features.mean(dim=1, keepdim=True)  # (B, 1, D)

        # Predict target embeddings by combining context and positional info
        predictions = context_mean + target_pos_emb  # (B, T_target, D)
        predictions = self.output_proj(predictions)

        return predictions


class JEPAVideoModel(nn.Module):
    """Complete JEPA video embedding model with student-teacher architecture.

    Implements JEPA-like self-supervised learning:
    - Student encoder processes frames and makes predictions via temporal predictor
    - Teacher encoder (EMA of student) provides targets
    - No contrastive negatives needed

    Args:
        student_encoder: Student frame encoder
        teacher_encoder: Teacher frame encoder (will be updated via EMA)
        predictor: Temporal predictor
        ema_decay: EMA decay rate for teacher updates
    """

    def __init__(
        self,
        student_encoder: FrameEncoder,
        teacher_encoder: FrameEncoder,
        predictor: TemporalPredictor,
        ema_decay: float = 0.999
    ):
        super().__init__()
        self.student_encoder = student_encoder
        self.teacher_encoder = teacher_encoder
        self.predictor = predictor
        self.ema_decay = ema_decay

        # Initialize teacher as copy of student
        self._initialize_teacher()

        # Teacher should not require gradients
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

    def _initialize_teacher(self):
        """Initialize teacher encoder as copy of student encoder."""
        for teacher_param, student_param in zip(
            self.teacher_encoder.parameters(),
            self.student_encoder.parameters()
        ):
            teacher_param.data.copy_(student_param.data)

    @torch.no_grad()
    def update_teacher(self):
        """Update teacher encoder using exponential moving average of student."""
        for teacher_param, student_param in zip(
            self.teacher_encoder.parameters(),
            self.student_encoder.parameters()
        ):
            teacher_param.data.mul_(self.ema_decay).add_(
                student_param.data, alpha=1 - self.ema_decay
            )

    def forward(
        self,
        frames: torch.Tensor,
        context_frames: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: Video frames, uint8 tensor of shape (B, T, 3, H, W)
            context_frames: Number of frames to use as context (rest are targets)

        Returns:
            predictions: Predicted embeddings for target frames, shape (B, T_target, D)
            targets: Target embeddings from teacher, shape (B, T_target, D)
        """
        B, T, C, H, W = frames.shape

        # Ensure we have enough frames
        assert T > context_frames, f"Need more than {context_frames} frames, got {T}"

        # Randomly sample context frame indices for each batch element
        context_indices = torch.stack([
            torch.randperm(T, device=frames.device)[:context_frames].sort()[0]
            for _ in range(B)
        ])  # (B, context_frames)

        # Create target indices (all frames not in context)
        all_indices = torch.arange(T, device=frames.device).unsqueeze(0).expand(B, -1)  # (B, T)
        target_mask = torch.ones(B, T, dtype=torch.bool, device=frames.device)
        target_mask.scatter_(1, context_indices, False)
        target_indices = all_indices[target_mask].view(B, T - context_frames)  # (B, T-context_frames)

        # Gather context and target frames
        context_indices_expanded = context_indices.view(B, context_frames, 1, 1, 1).expand(B, context_frames, C, H, W)
        target_indices_expanded = target_indices.view(B, T - context_frames, 1, 1, 1).expand(B, T - context_frames, C, H, W)

        context_frames_tensor = torch.gather(frames, 1, context_indices_expanded)  # (B, context_frames, C, H, W)
        target_frames_tensor = torch.gather(frames, 1, target_indices_expanded)   # (B, T-context_frames, C, H, W)

        # Flatten batch and time dimensions for encoding
        context_flat = context_frames_tensor.reshape(B * context_frames, C, H, W)
        target_flat = target_frames_tensor.reshape(B * (T - context_frames), C, H, W)

        # Encode context frames with student
        context_embeddings = self.student_encoder(context_flat)  # (B*context_frames, D)
        context_embeddings = context_embeddings.view(B, context_frames, -1)  # (B, context_frames, D)

        # Encode target frames with teacher (no gradients)
        with torch.no_grad():
            target_embeddings = self.teacher_encoder(target_flat)  # (B*(T-context_frames), D)
            target_embeddings = target_embeddings.view(B, T - context_frames, -1)  # (B, T-context_frames, D)

        # Predict target embeddings using predictor
        predictions = self.predictor(context_embeddings, target_indices)  # (B, T-context_frames, D)

        return predictions, target_embeddings

    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode all frames of a video using student encoder.

        Args:
            frames: Video frames, uint8 tensor of shape (T, 3, H, W) or (B, T, 3, H, W)

        Returns:
            embeddings: Frame embeddings, shape (T, D) or (B, T, D)
        """
        if frames.dim() == 4:
            # Single video: (T, C, H, W)
            T, C, H, W = frames.shape
            embeddings = self.student_encoder(frames)  # (T, D)
            return embeddings
        elif frames.dim() == 5:
            # Batch of videos: (B, T, C, H, W)
            B, T, C, H, W = frames.shape
            frames_flat = frames.view(B * T, C, H, W)
            embeddings = self.student_encoder(frames_flat)  # (B*T, D)
            embeddings = embeddings.view(B, T, -1)  # (B, T, D)
            return embeddings
        else:
            raise ValueError(f"Expected 4D or 5D input, got {frames.dim()}D")

