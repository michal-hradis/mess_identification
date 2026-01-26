"""Loss functions for identity learning."""
import torch
from functools import partial
from pytorch_metric_learning import losses


def contrastive_loss(emb, labels, temperature=1.0, old_emb=None, old_labels=None):
    """
    Contrastive loss with optional batch history.

    Args:
        emb: Embeddings tensor of shape (batch_size, emb_dim)
        labels: Label tensor of shape (batch_size,)
        temperature: Temperature parameter for softmax
        old_emb: Optional embeddings from previous batches
        old_labels: Optional labels from previous batches

    Returns:
        Scalar loss value
    """
    sim = (emb @ emb.t()) / temperature

    # mask the same images on the main diagonal
    sim.fill_diagonal_(-1e20)

    # maximum value for stable exp --- without the main diagonal which is not used
    with torch.no_grad():
        max_val = torch.amax(sim, dim=1, keepdim=True)
    sim = torch.exp(sim - max_val)

    if old_emb is not None:
        # mask old positives
        sim_old = (emb @ old_emb.t()) / temperature
        sim_old[labels.reshape(-1, 1) == old_labels.reshape(1, -1)] = -1e20
        sim_old = torch.exp(sim_old - max_val)

    with torch.no_grad():
        pos = labels.reshape(-1, 1) == labels.reshape(1, -1)
        neg = labels.reshape(-1, 1) != labels.reshape(1, -1)
        pos.fill_diagonal_(0)

    numerator = sim * pos
    if old_emb is not None:
        denominator = torch.sum(sim * neg, dim=1, keepdim=True) + torch.sum(sim_old, dim=1, keepdim=True) + numerator
    else:
        denominator = torch.sum(sim * neg, dim=1, keepdim=True) + numerator
    loss = -torch.log(numerator + 1e-20) + torch.log(denominator + 1e-20) * pos
    loss = torch.sum(loss) / torch.sum(pos)

    return loss


def get_loss_function(loss_type, args, dataset_size, device):
    """
    Factory function to create loss functions and their optimizers.

    Args:
        loss_type: Type of loss ('normalized_softmax', 'arcface', 'xent')
        args: Command line arguments
        dataset_size: Size of the training dataset
        device: Device to place loss on

    Returns:
        Tuple of (loss_function, loss_optimizer or None)
    """
    size_multiplier = 1
    num_classes = dataset_size // size_multiplier + 1

    if loss_type == 'normalized_softmax':
        loss_fce = losses.NormalizedSoftmaxLoss(
            num_classes,
            args.emb_dim,
            temperature=args.temperature
        ).to(device)
        loss_optimizer = torch.optim.AdamW(
            loss_fce.parameters(),
            lr=args.loss_learning_rate,
            weight_decay=args.loss_weight_decay
        )
        return loss_fce, loss_optimizer

    elif loss_type == 'arcface':
        loss_fce = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=args.emb_dim
        ).to(device)
        loss_optimizer = torch.optim.AdamW(
            loss_fce.parameters(),
            lr=args.loss_learning_rate,
            weight_decay=args.loss_weight_decay
        )
        return loss_fce, loss_optimizer

    elif loss_type == 'xent':
        loss_fce = partial(contrastive_loss, temperature=args.temperature)
        return loss_fce, None

    else:
        raise ValueError(f'Unknown loss function "{loss_type}". Available options are: normalized_softmax, arcface, xent.')

