"""
Test script for Domain Adaptation module.
Verifies gradient reversal and domain classifier functionality.
"""
import torch
import torch.nn as nn
from code.domain_adaptation import (
    GradientReversalLayer,
    DomainClassifier,
    EmbeddingModelWithDomainAdaptation
)


def test_gradient_reversal():
    """Test that gradients are reversed correctly"""
    print("Testing Gradient Reversal Layer...")

    grl = GradientReversalLayer(lambda_=1.0)

    # Create input that requires grad
    x = torch.randn(4, 128, requires_grad=True)

    # Forward pass
    y = grl(x)

    # Backward pass
    loss = y.sum()
    loss.backward()

    # Check gradient - should be negative (reversed)
    assert x.grad is not None, "Gradient not computed"
    expected_grad = -torch.ones_like(x)
    assert torch.allclose(x.grad, expected_grad), "Gradient not reversed correctly"

    print("✓ Gradient Reversal Layer works correctly")


def test_domain_classifier():
    """Test domain classifier forward pass"""
    print("\nTesting Domain Classifier...")

    batch_size = 16
    emb_dim = 128
    num_domains = 10

    classifier = DomainClassifier(
        input_dim=emb_dim,
        num_domains=num_domains,
        hidden_dims=[256, 128],
        dropout=0.5
    )

    # Test forward pass
    embeddings = torch.randn(batch_size, emb_dim)
    logits = classifier(embeddings)

    assert logits.shape == (batch_size, num_domains), f"Wrong output shape: {logits.shape}"

    print(f"✓ Domain Classifier output shape: {logits.shape}")

    # Test lambda update
    classifier.set_lambda(2.0)
    assert classifier.grl.lambda_ == 2.0, "Lambda not updated correctly"
    print("✓ Lambda update works correctly")


def test_embedding_model_wrapper():
    """Test the full wrapper with domain adaptation"""
    print("\nTesting EmbeddingModelWithDomainAdaptation...")

    # Create a simple mock embedding model
    class MockEmbeddingModel(nn.Module):
        def __init__(self, emb_dim=128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            self.decoder = nn.Linear(64, emb_dim)
            self.decoder.emb_dim = emb_dim  # Add emb_dim attribute

        def forward(self, x):
            x = x.float() / 255.0 if x.dtype == torch.uint8 else x
            x = self.encoder(x)
            return self.decoder(x)

    batch_size = 8
    num_domains = 5
    emb_dim = 128

    base_model = MockEmbeddingModel(emb_dim=emb_dim)

    # Test with domain adaptation enabled
    model = EmbeddingModelWithDomainAdaptation(
        base_model,
        num_domains=num_domains,
        domain_hidden_dims=[256, 128],
        domain_dropout=0.5
    )

    # Test forward pass without domain logits
    images = torch.randint(0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8)
    embeddings = model(images, return_domain_logits=False)
    assert embeddings.shape == (batch_size, emb_dim), f"Wrong embedding shape: {embeddings.shape}"
    print(f"✓ Embeddings shape (no domain logits): {embeddings.shape}")

    # Test forward pass with domain logits
    embeddings, domain_logits = model(images, return_domain_logits=True)
    assert embeddings.shape == (batch_size, emb_dim), f"Wrong embedding shape: {embeddings.shape}"
    assert domain_logits.shape == (batch_size, num_domains), f"Wrong domain logits shape: {domain_logits.shape}"
    print(f"✓ Embeddings shape (with domain logits): {embeddings.shape}")
    print(f"✓ Domain logits shape: {domain_logits.shape}")

    # Test get_embedding_model
    extracted_model = model.get_embedding_model()
    assert extracted_model is base_model, "Extracted model is not the base model"
    print("✓ Base model extraction works correctly")

    # Test without domain adaptation
    model_no_da = EmbeddingModelWithDomainAdaptation(base_model, num_domains=None)
    embeddings = model_no_da(images)
    assert embeddings.shape == (batch_size, emb_dim), f"Wrong embedding shape: {embeddings.shape}"
    print(f"✓ Model without domain adaptation works correctly")


def test_gradient_flow():
    """Test that gradients flow correctly through the system"""
    print("\nTesting Gradient Flow...")

    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 64)
            self.decoder = nn.Linear(64, 32)
            self.decoder.emb_dim = 32

        def forward(self, x):
            return self.decoder(torch.relu(self.fc(x)))

    base_model = SimpleModel()
    model = EmbeddingModelWithDomainAdaptation(
        base_model,
        num_domains=5,
        domain_hidden_dims=[128],
        domain_dropout=0.0
    )

    # Forward pass
    x = torch.randn(4, 10)
    embeddings, domain_logits = model(x, return_domain_logits=True)

    # Create dummy labels and compute losses
    identity_labels = torch.tensor([0, 1, 0, 1])
    domain_labels = torch.tensor([0, 1, 2, 3])

    # Embedding loss (simple MSE for testing)
    emb_loss = ((embeddings - torch.randn_like(embeddings)) ** 2).mean()

    # Domain loss
    domain_loss = nn.functional.cross_entropy(domain_logits, domain_labels)

    # Total loss
    total_loss = emb_loss + 0.1 * domain_loss

    # Backward
    total_loss.backward()

    # Check that gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"

    print("✓ Gradients flow correctly through the model")


if __name__ == "__main__":
    print("=" * 60)
    print("Domain Adaptation Module Tests")
    print("=" * 60)

    try:
        test_gradient_reversal()
        test_domain_classifier()
        test_embedding_model_wrapper()
        test_gradient_flow()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

