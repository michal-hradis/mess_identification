"""Trainer class for identity learning."""
import logging
from collections import defaultdict
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


class IdentityTrainer:
    """Handles training loop and optimization for identity learning."""

    def __init__(self, model, loss_fn, optimizer, args, device, loss_optimizer=None):
        """
        Initialize trainer.

        Args:
            model: Model to train
            loss_fn: Loss function
            optimizer: Model optimizer
            args: Command line arguments
            device: Device to train on
            loss_optimizer: Optional optimizer for loss function parameters
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.loss_optimizer = loss_optimizer
        self.args = args
        self.device = device
        self.loss_history = defaultdict(list)
        self.iteration = args.start_iteration

        # Batch history for contrastive learning
        self.embed_history = []
        self.label_history = []

        # Test tracking
        self.test_similarities = []
        self.test_labels = []

        # GPU augmentation
        self.aug_gpu = None
        if args.gpu_augmentation:
            from common import augmentations_gpu
            self.aug_gpu = augmentations_gpu.GPU_AUGMENTATIONS[args.gpu_augmentation]
            logging.info(f'Using GPU augmentation: {self.aug_gpu}')
            self.aug_gpu = self.aug_gpu.to(device)

    def prepare_batch(self, batch_data):
        """
        Prepare batch data for training.

        Args:
            batch_data: Dictionary containing batch data

        Returns:
            Tuple of (images, labels, video_ids)
        """
        # Handle dictionary batch data
        if 'image' in batch_data:
            # Single image mode
            images = batch_data['image'].to(self.device)
            labels = batch_data['label'].to(self.device)
            video_ids = batch_data['video_id'].to(self.device)
        else:
            # Two image mode
            images1 = batch_data['image1'].to(self.device)
            images2 = batch_data['image2'].to(self.device)
            images = torch.cat([images1, images2], dim=0)
            labels = torch.cat([batch_data['label'], batch_data['label']]).to(self.device)
            video_ids = torch.cat([batch_data['video_id'], batch_data['video_id']]).to(self.device)

        # Apply GPU augmentation if enabled
        if self.aug_gpu is not None:
            images = images.float() / 255.0

        return images, labels, video_ids

    def adjust_learning_rate(self):
        """Adjust learning rate during warmup period."""
        if self.iteration - self.args.start_iteration <= self.args.warmup_iterations:
            warmup_progress = min(self.iteration - self.args.start_iteration, self.args.warmup_iterations)
            lr = (warmup_progress / self.args.warmup_iterations) ** 1.5 * self.args.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def compute_loss(self, embedding, labels, video_ids, domain_logits=None):
        """
        Compute total loss with all components.

        Args:
            embedding: Model embeddings
            labels: Labels tensor
            video_ids: Video IDs tensor
            domain_logits: Optional domain classification logits

        Returns:
            Total loss
        """
        # Main embedding loss
        loss = self.loss_fn(embedding, labels)
        self.loss_history["main"].append(loss.item())

        # Add domain adaptation loss if enabled
        if self.args.use_domain_adaptation and domain_logits is not None:
            domain_loss = self.args.domain_loss_weight * torch.nn.functional.cross_entropy(
                domain_logits, video_ids
            )
            self.loss_history["domain"].append(domain_loss.item())
            loss = loss + domain_loss

        # Add embedding regularization
        if self.args.emb_reg_weight > 0:
            emb_reg_loss = self.args.emb_reg_weight * torch.mean(embedding ** 2)
            self.loss_history["emb_reg"].append(emb_reg_loss.item())
            loss = loss + emb_reg_loss

        self.loss_history["full"].append(loss.item())
        return loss

    def train_step(self, batch_data):
        """
        Perform a single training step.

        Args:
            batch_data: Dictionary containing batch data

        Returns:
            Tuple of (loss, embedding, images)
        """
        # Prepare batch
        images, labels, video_ids = self.prepare_batch(batch_data)

        # Adjust learning rate during warmup
        self.adjust_learning_rate()

        # Forward pass
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Get embeddings and optionally domain logits
            if self.args.use_domain_adaptation:
                embedding, domain_logits = self.model(images, return_domain_logits=True)
            else:
                embedding = self.model(images)
                domain_logits = None

            # Compute loss
            loss = self.compute_loss(embedding, labels, video_ids, domain_logits)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        if self.loss_optimizer is not None:
            self.loss_optimizer.step()
            self.loss_optimizer.zero_grad()

        self.optimizer.zero_grad()

        # Update batch history if enabled
        if self.args.batch_history:
            with torch.no_grad():
                self.embed_history.append(embedding.detach())
                self.label_history.append(labels.detach())
            self.embed_history = self.embed_history[-self.args.batch_history:]
            self.label_history = self.label_history[-self.args.batch_history:]

        return loss.item(), embedding, images

    def should_evaluate(self):
        """Check if evaluation should be performed at current iteration."""
        return self.iteration % self.args.view_step == 0 and self.iteration > self.args.start_iteration

    def is_test_tracking(self):
        """Check if we should track test data for this iteration."""
        to_test = 20
        return self.iteration % self.args.view_step > self.args.view_step - to_test

    def track_test_data(self, embedding, labels):
        """Track test similarities and labels for visualization."""
        self.test_labels.append(labels.cpu().numpy())
        self.test_similarities.append(
            torch.mm(embedding, embedding.t()).detach().cpu().numpy()
        )

    def save_checkpoint(self):
        """Save model checkpoint."""
        # Save checkpoint - only base model without domain adaptation head
        if self.args.use_domain_adaptation:
            checkpoint_state = self.model.get_embedding_model().state_dict()
        else:
            checkpoint_state = self.model.state_dict()

        checkpoint_path = f'cp-{self.iteration:07d}.img.ckpt'
        torch.save(checkpoint_state, checkpoint_path)
        logging.info(f'Saved checkpoint: {checkpoint_path}')

    def export_model(self, images):
        """
        Export model to TorchScript format.

        Args:
            images: Sample images for tracing
        """
        self.model.eval()
        export_input = torch.randint(
            0, 256,
            (128, 3, images.shape[2], images.shape[3]),
            dtype=torch.uint8
        ).to(images.device)
        traced_model = torch.jit.trace(self.model, export_input)
        export_path = f'{self.args.name}_{self.iteration:07d}.pt'
        torch.jit.save(traced_model, export_path)
        self.model.train()
        logging.info(f'Exported model: {export_path}')

    def plot_similarity_heatmap(self):
        """Plot and save similarity heatmap from recent test data."""
        if not self.test_similarities:
            return

        print('SIMILARITIES', self.test_similarities[-1].min(), self.test_similarities[-1].max())
        fig, ax = plt.subplots()
        heatmap = ax.imshow(self.test_similarities[-1])
        plt.colorbar(heatmap)
        plt.savefig(f'cp-{self.iteration:07d}.png')
        plt.close('all')

    def get_loss_summary(self):
        """Get summary string of average losses."""
        return ' '.join([f'{k}:{np.mean(v):0.5f}' for k, v in self.loss_history.items()])

    def reset_test_tracking(self):
        """Reset test tracking data."""
        self.test_similarities = []
        self.test_labels = []
        self.loss_history = defaultdict(list)

    def increment_iteration(self):
        """Increment iteration counter."""
        self.iteration += 1

    def is_max_iteration_reached(self):
        """Check if maximum iteration is reached."""
        return self.iteration > self.args.max_iteration


def tile_images(images, step=16):
    """
    Tile images into a grid.

    Args:
        images: List of images
        step: Number of images per row

    Returns:
        Tiled image as numpy array
    """
    lines = [np.concatenate(images[i:i+step], axis=1) for i in range(0, len(images), step)]
    return np.concatenate(lines, axis=0)


def init_central_loss_embeddings(model, dataset, loss_fce, device, batch_size=32):
    """
    Initialize loss function embeddings with dataset centroids.

    Args:
        model: Model to use for computing embeddings
        dataset: Dataset to initialize from
        loss_fce: Loss function with learnable parameters
        device: Device to run on
        batch_size: Batch size for processing
    """
    from tqdm import tqdm

    with torch.no_grad():
        # Compute embeddings for all images and store them in the loss function
        all_embeddings = []
        all_labels = []
        batch = []
        model.eval()

        for i in tqdm(range(dataset.image_count())):
            img, label, video_id = dataset.get_image(i)
            batch.append(img)
            all_labels.append(label)
            if len(batch) >= batch_size:
                batch = np.stack(batch, axis=0)
                batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
                res = model(batch)
                all_embeddings.append(res)
                batch = []

        if batch:
            batch = np.stack(batch, axis=0)
            batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
            res = model(batch)
            all_embeddings.append(res)

        # average embeddings of each unique label
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = np.asarray(all_labels)
        unique_labels = set(all_labels)
        wMat = list(loss_fce.parameters())[0]

        print(wMat.shape, len(list(loss_fce.parameters())), np.max(unique_labels))
        print(sorted(unique_labels))

        for label in unique_labels:
            idx = np.where(all_labels == label)[0]
            embeddings = all_embeddings[idx]
            embeddings = torch.mean(embeddings, dim=0)
            # set the embedding in the loss function for the label - write into the weight matrix
            wMat[:, label] = embeddings

