"""Simple test script to verify video embedding components."""
import torch
import sys
sys.path.insert(0, '..')

from video_embeddings.video_dataset import VideoDataset
from video_embeddings.jepa_model import FrameEncoder, TemporalPredictor, JEPAVideoModel
from video_embeddings.lightning_module import VideoEmbeddingModule, JEPALoss
from common.nets_pretrained import PretrainedEncoder


def test_frame_encoder():
    """Test FrameEncoder."""
    print("Testing FrameEncoder...")

    # Create base encoder
    base_encoder = PretrainedEncoder(
        name='resnet34',
        depth=5,
        weights=None,  # No pretrained weights for testing
        in_channels=3
    )

    # Create frame encoder
    encoder = FrameEncoder(
        base_encoder=base_encoder,
        embedding_dim=256,
        normalize=True
    )

    # Test with uint8 input
    frames = torch.randint(0, 256, (4, 3, 224, 224), dtype=torch.uint8)
    embeddings = encoder(frames)

    assert embeddings.shape == (4, 256), f"Expected (4, 256), got {embeddings.shape}"
    assert torch.allclose(torch.norm(embeddings, p=2, dim=1), torch.ones(4), atol=1e-5), "Embeddings should be normalized"

    print("✓ FrameEncoder test passed!")


def test_temporal_predictor():
    """Test TemporalPredictor."""
    print("Testing TemporalPredictor...")

    predictor = TemporalPredictor(
        embedding_dim=256,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
        max_frames=16
    )

    # Test input
    embeddings = torch.randn(2, 8, 256)
    predictions = predictor(embeddings)

    assert predictions.shape == (2, 8, 256), f"Expected (2, 8, 256), got {predictions.shape}"

    print("✓ TemporalPredictor test passed!")


def test_jepa_model():
    """Test JEPAVideoModel."""
    print("Testing JEPAVideoModel...")

    # Create encoders
    base_encoder_student = PretrainedEncoder(
        name='resnet34',
        depth=5,
        weights=None,
        in_channels=3
    )

    base_encoder_teacher = PretrainedEncoder(
        name='resnet34',
        depth=5,
        weights=None,
        in_channels=3
    )

    student_encoder = FrameEncoder(
        base_encoder=base_encoder_student,
        embedding_dim=128,
        normalize=True
    )

    teacher_encoder = FrameEncoder(
        base_encoder=base_encoder_teacher,
        embedding_dim=128,
        normalize=True
    )

    predictor = TemporalPredictor(
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )

    model = JEPAVideoModel(
        student_encoder=student_encoder,
        teacher_encoder=teacher_encoder,
        predictor=predictor,
        ema_decay=0.999
    )

    # Test forward pass
    frames = torch.randint(0, 256, (2, 8, 3, 224, 224), dtype=torch.uint8)
    context_frames = 4

    student_preds, teacher_targets = model(frames, context_frames=context_frames)

    expected_shape = (2, 8 - context_frames, 128)
    assert student_preds.shape == expected_shape, f"Expected {expected_shape}, got {student_preds.shape}"
    assert teacher_targets.shape == expected_shape, f"Expected {expected_shape}, got {teacher_targets.shape}"

    # Test teacher update
    model.update_teacher()

    print("✓ JEPAVideoModel test passed!")


def test_jepa_loss():
    """Test JEPALoss."""
    print("Testing JEPALoss...")

    loss_fn = JEPALoss(loss_type='smooth_l1')

    predictions = torch.randn(2, 4, 128)
    targets = torch.randn(2, 4, 128)

    loss = loss_fn(predictions, targets)

    assert loss.ndim == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("✓ JEPALoss test passed!")


def test_lightning_module():
    """Test VideoEmbeddingModule."""
    print("Testing VideoEmbeddingModule...")

    # Create model
    base_encoder_student = PretrainedEncoder(
        name='resnet34',
        depth=5,
        weights=None,
        in_channels=3
    )

    base_encoder_teacher = PretrainedEncoder(
        name='resnet34',
        depth=5,
        weights=None,
        in_channels=3
    )

    student_encoder = FrameEncoder(
        base_encoder=base_encoder_student,
        embedding_dim=128,
        normalize=True
    )

    teacher_encoder = FrameEncoder(
        base_encoder=base_encoder_teacher,
        embedding_dim=128,
        normalize=True
    )

    predictor = TemporalPredictor(
        embedding_dim=128,
        num_heads=4,
        num_layers=2
    )

    jepa_model = JEPAVideoModel(
        student_encoder=student_encoder,
        teacher_encoder=teacher_encoder,
        predictor=predictor
    )

    pl_module = VideoEmbeddingModule(
        model=jepa_model,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
        context_frames=4,
        loss_type='smooth_l1'
    )

    # Test forward pass
    batch = {
        'frames': torch.randint(0, 256, (2, 8, 3, 224, 224), dtype=torch.uint8),
        'video_id': torch.tensor([0, 1])
    }

    predictions, targets = pl_module(batch['frames'])

    assert predictions.shape[0] == 2, "Batch size should be 2"
    assert predictions.shape[2] == 128, "Embedding dim should be 128"

    print("✓ VideoEmbeddingModule test passed!")


if __name__ == '__main__':
    print("Running video embedding tests...\n")

    test_frame_encoder()
    test_temporal_predictor()
    test_jepa_model()
    test_jepa_loss()
    test_lightning_module()

    print("\n✓ All tests passed!")
