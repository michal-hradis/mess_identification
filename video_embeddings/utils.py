"""Utility functions for video embeddings."""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


def export_encoder(
    checkpoint_path: str,
    output_path: str,
    trace: bool = True
):
    """
    Export trained encoder from Lightning checkpoint.

    Args:
        checkpoint_path: Path to Lightning checkpoint
        output_path: Path to save exported model
        trace: Whether to use TorchScript tracing (vs scripting)
    """
    from lightning_module import VideoEmbeddingModule

    # Load checkpoint
    pl_module = VideoEmbeddingModule.load_from_checkpoint(checkpoint_path)

    # Extract student encoder
    encoder = pl_module.model.student_encoder
    encoder.eval()

    # Export
    if trace:
        dummy_input = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8)
        traced_model = torch.jit.trace(encoder, dummy_input)
        traced_model.save(output_path)
        print(f"Traced encoder saved to {output_path}")
    else:
        torch.save(encoder.state_dict(), output_path)
        print(f"Encoder state dict saved to {output_path}")


def load_encoder_for_inference(
    checkpoint_path: str,
    device: str = 'cuda'
) -> nn.Module:
    """
    Load trained encoder for inference.

    Args:
        checkpoint_path: Path to Lightning checkpoint or state dict
        device: Device to load model on

    Returns:
        encoder: Loaded encoder model
    """
    from lightning_module import VideoEmbeddingModule

    # Try loading as Lightning checkpoint
    try:
        pl_module = VideoEmbeddingModule.load_from_checkpoint(checkpoint_path)
        encoder = pl_module.model.student_encoder
    except:
        # Try loading as TorchScript
        try:
            encoder = torch.jit.load(checkpoint_path)
        except:
            raise ValueError(f"Could not load model from {checkpoint_path}")

    encoder = encoder.to(device)
    encoder.eval()

    return encoder


def extract_video_embedding(
    encoder: nn.Module,
    frames: torch.Tensor,
    aggregate: str = 'mean'
) -> torch.Tensor:
    """
    Extract video-level embedding from frames.

    Args:
        encoder: Frame encoder model
        frames: (T, C, H, W) or (B, T, C, H, W) frame tensor
        aggregate: Aggregation method ('mean', 'max', 'first', 'last')

    Returns:
        embedding: (D,) or (B, D) video embedding
    """
    if frames.ndim == 4:
        # Single video: (T, C, H, W)
        frames = frames.unsqueeze(0)  # (1, T, C, H, W)
        squeeze_output = True
    else:
        squeeze_output = False

    B, T, C, H, W = frames.shape

    # Flatten batch and time
    frames_flat = frames.view(B * T, C, H, W)

    # Encode frames
    with torch.no_grad():
        embeddings_flat = encoder(frames_flat)  # (B*T, D)

    # Reshape
    D = embeddings_flat.shape[1]
    embeddings = embeddings_flat.view(B, T, D)

    # Aggregate over time
    if aggregate == 'mean':
        video_embedding = embeddings.mean(dim=1)
    elif aggregate == 'max':
        video_embedding = embeddings.max(dim=1)[0]
    elif aggregate == 'first':
        video_embedding = embeddings[:, 0]
    elif aggregate == 'last':
        video_embedding = embeddings[:, -1]
    else:
        raise ValueError(f"Unknown aggregation: {aggregate}")

    if squeeze_output:
        video_embedding = video_embedding.squeeze(0)

    return video_embedding


class VideoEncoder(nn.Module):
    """
    Wrapper for frame encoder that processes videos.
    Useful for inference and downstream tasks.
    """

    def __init__(
        self,
        frame_encoder: nn.Module,
        aggregate: str = 'mean',
        normalize: bool = True
    ):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.aggregate = aggregate
        self.normalize = normalize

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) video frames

        Returns:
            embeddings: (B, D) video embeddings
        """
        B, T, C, H, W = frames.shape

        # Flatten
        frames_flat = frames.view(B * T, C, H, W)

        # Encode
        embeddings_flat = self.frame_encoder(frames_flat)

        # Reshape
        D = embeddings_flat.shape[1]
        embeddings = embeddings_flat.view(B, T, D)

        # Aggregate
        if self.aggregate == 'mean':
            video_embedding = embeddings.mean(dim=1)
        elif self.aggregate == 'max':
            video_embedding = embeddings.max(dim=1)[0]
        elif self.aggregate == 'first':
            video_embedding = embeddings[:, 0]
        elif self.aggregate == 'last':
            video_embedding = embeddings[:, -1]

        # Normalize
        if self.normalize:
            video_embedding = torch.nn.functional.normalize(video_embedding, p=2, dim=-1)

        return video_embedding


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Export trained encoder')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Lightning checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for exported model')
    parser.add_argument('--no-trace', action='store_true',
                        help='Save state dict instead of tracing')

    args = parser.parse_args()

    export_encoder(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        trace=not args.no_trace
    )
