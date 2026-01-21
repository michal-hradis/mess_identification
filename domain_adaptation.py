"""
Domain Adaptation components for improving model robustness.
Includes Gradient Reversal Layer and Domain Classifier.
"""
import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    "Unsupervised Domain Adaptation by Backpropagation" (Ganin & Lempitsky, 2015)

    During forward pass, it acts as an identity function.
    During backward pass, it reverses the gradient by multiplying by -lambda.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer wrapper.
    """
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainClassifier(nn.Module):
    """
    MLP-based domain classifier for video_id classification.
    Connected after a Gradient Reversal Layer to encourage domain-invariant features.
    """
    def __init__(self, input_dim, num_domains, hidden_dims=[256, 128], dropout=0.5):
        """
        Args:
            input_dim: Dimension of input embeddings
            num_domains: Number of video/domain classes to predict
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(DomainClassifier, self).__init__()

        self.grl = GradientReversalLayer(lambda_=1.0)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_domains))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input embeddings (batch_size, input_dim)
        Returns:
            logits: Domain classification logits (batch_size, num_domains)
        """
        # Apply gradient reversal
        x = self.grl(x)
        # Classify domain
        logits = self.classifier(x)
        return logits

    def set_lambda(self, lambda_):
        """Update the gradient reversal strength"""
        self.grl.lambda_ = lambda_


class EmbeddingModelWithDomainAdaptation(nn.Module):
    """
    Wrapper for EmbeddingModel that adds optional domain adaptation head.
    """
    def __init__(self, embedding_model, num_domains=None, domain_hidden_dims=[256, 128],
                 domain_dropout=0.5):
        """
        Args:
            embedding_model: The base embedding model
            num_domains: Number of domains (video_ids). If None, domain adaptation is disabled.
            domain_hidden_dims: Hidden dimensions for domain classifier MLP
            domain_dropout: Dropout for domain classifier
        """
        super(EmbeddingModelWithDomainAdaptation, self).__init__()

        self.embedding_model = embedding_model
        self.use_domain_adaptation = num_domains is not None and num_domains > 0

        if self.use_domain_adaptation:
            # Get embedding dimension from the model
            emb_dim = embedding_model.decoder.emb_dim
            self.domain_classifier = DomainClassifier(
                input_dim=emb_dim,
                num_domains=num_domains,
                hidden_dims=domain_hidden_dims,
                dropout=domain_dropout
            )
        else:
            self.domain_classifier = None

    def forward(self, x, return_domain_logits=False):
        """
        Args:
            x: Input images
            return_domain_logits: If True, also return domain classification logits
        Returns:
            embeddings: Image embeddings
            domain_logits: (optional) Domain classification logits
        """
        embeddings = self.embedding_model(x)

        if return_domain_logits and self.use_domain_adaptation:
            domain_logits = self.domain_classifier(embeddings)
            return embeddings, domain_logits
        else:
            return embeddings

    def set_grl_lambda(self, lambda_):
        """Update gradient reversal strength"""
        if self.use_domain_adaptation:
            self.domain_classifier.set_lambda(lambda_)

    def get_embedding_model(self):
        """Get the base embedding model without domain adaptation head"""
        return self.embedding_model

