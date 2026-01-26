"""Evaluation utilities for retrieval tasks."""
import time
import logging
import numpy as np
import cv2
import torch
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class RetrievalEvaluator:
    """Handles model evaluation for retrieval tasks."""

    def __init__(self, device, batch_size=32):
        """
        Initialize evaluator.

        Args:
            device: Device to run evaluation on
            batch_size: Batch size for computing embeddings
        """
        self.device = device
        self.batch_size = batch_size

    def compute_embeddings(self, model, images):
        """
        Compute embeddings for a list of images.

        Args:
            model: Model to use for computing embeddings
            images: List of images (numpy arrays)

        Returns:
            Tensor of embeddings
        """
        embeddings = []
        img_to_process = list(images)
        model.eval()

        with torch.no_grad():
            while img_to_process:
                batch = img_to_process[:self.batch_size]
                img_to_process = img_to_process[self.batch_size:]
                batch = np.stack(batch, axis=0)
                batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
                res = model(batch)
                embeddings.append(res)

        model.train()
        return torch.cat(embeddings, dim=0)

    def evaluate_retrieval(self, model, dataset, max_img=2000, query_vis_count=16, result_vis_count=20):
        """
        Evaluate retrieval performance on a dataset.

        Args:
            model: Model to evaluate
            dataset: Dataset to evaluate on
            max_img: Maximum number of images to use
            query_vis_count: Number of queries to visualize
            result_vis_count: Number of results per query to visualize

        Returns:
            Dictionary with metrics: auc, mean_auc, mean_ap, fpr, tpr, thresholds,
            all_images, all_labels, similarities
        """
        t_start = time.time()
        all_images = []
        all_labels = []

        seed = 0
        identity_count = len(dataset)
        identity_permutation = np.random.RandomState(seed).permutation(identity_count)

        for i in identity_permutation:
            images = dataset.get_all_id_images(i)

            # select random images from the identity, but this has to be repeatable
            if len(images) > 6:
                selected = np.random.RandomState(seed).permutation(len(images))[:6]
                images = [images[i] for i in selected]

            all_images.extend(images)
            all_labels.extend([i] * len(images))
            if len(all_images) > max_img:
                break

        # Compute embeddings
        embeddings = self.compute_embeddings(model, all_images)
        similarities = embeddings @ embeddings.t()
        similarities = similarities.cpu().numpy()
        all_labels = np.asarray(all_labels)

        logging.info(f'SIMILARITIES {similarities.min()} {similarities.max()}, {similarities.shape}')

        # Compute metrics
        scores = []
        labels = []
        ap_list = []
        auc_list = []

        for i in range(similarities.shape[0]):
            query_sim = similarities[i]
            query_labels = all_labels[i] == all_labels
            # remove the query image
            query_labels[i] = False
            query_sim[i] = -1e20

            if np.any(query_labels):
                auc_list.append(roc_auc_score(query_labels, query_sim))
                ap_list.append(average_precision_score(query_labels, query_sim))

            scores.append(query_sim)
            labels.append(query_labels)

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
        mean_ap = np.mean(ap_list)
        mean_auc = np.mean(auc_list)
        global_auc = roc_auc_score(labels, scores)

        logging.info(f'Evaluation time: {time.time() - t_start:.1f}s')

        return {
            'auc': global_auc,
            'mean_auc': mean_auc,
            'mean_ap': mean_ap,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thr,
            'all_images': all_images,
            'all_labels': all_labels,
            'similarities': similarities
        }

    def create_result_collage(self, all_images, all_labels, similarities,
                             query_vis_count=16, result_vis_count=20):
        """
        Create visualization collage of retrieval results.

        Args:
            all_images: List of images
            all_labels: Array of labels
            similarities: Similarity matrix
            query_vis_count: Number of queries to visualize
            result_vis_count: Number of results per query

        Returns:
            Collage image as numpy array
        """
        collage_ids = np.linspace(0, len(all_images) - 1, query_vis_count).astype(np.int64)
        result_collage = []

        for i in collage_ids:
            query_sim = similarities[i]
            query_labels = all_labels[i] == all_labels
            # remove the query image
            query_labels[i] = False
            query_sim[i] = -1e20

            # Query image with blue border
            image = cv2.copyMakeBorder(all_images[i], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            row = [image]

            query_label = all_labels[i]
            best_ids = np.argsort(query_sim)[::-1][:result_vis_count]

            for idx in best_ids:
                image = all_images[idx]
                if all_labels[idx] == query_label:
                    # Correct match - green border
                    image = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 255, 0])
                else:
                    # Incorrect match - red border
                    image = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 0, 255])
                image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                row.append(image)

            result_collage.append(np.concatenate(row, axis=1))

        if result_collage:
            return np.concatenate(result_collage, axis=0)
        return None

    def plot_roc_curve(self, test_results, save_path):
        """
        Plot ROC curves for multiple test results.

        Args:
            test_results: Dictionary mapping names to result dictionaries
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots()
        for name, results in test_results.items():
            auc = results['auc']
            mean_auc = results['mean_auc']
            fpr = results['fpr']
            tpr = results['tpr']
            ax.semilogx(fpr, tpr, label=f'{name} AUC = {auc:.4f}, {mean_auc:.4f}')
        ax.legend()
        plt.xlim([0.000001, 1])
        plt.savefig(save_path)
        plt.close('all')


def test_simple(iteration, name, model, dataset, device, max_img, max_query_img):
    """
    Simple test function (legacy compatibility).

    Args:
        iteration: Current training iteration
        name: Name for saving outputs
        model: Model to evaluate
        dataset: Dataset to evaluate on
        device: Device to run on
        max_img: Maximum images to use
        max_query_img: Maximum query images

    Returns:
        AUC score
    """
    batch = []
    all_data = []
    all_labels = []
    print(max_img, len(dataset))

    with torch.no_grad():
        model = model.eval()
        for i in range(min(max_img, len(dataset))):
            data = dataset[i]
            if 'image' in data:
                # Single image mode
                batch.append(data['image'])
                all_labels.append(data['label'])
            else:
                # Two image mode
                batch.append(data['image1'])
                batch.append(data['image2'])
                all_labels.append(data['label'])
                all_labels.append(data['label'])
            if len(batch) >= 32:
                batch = torch.stack(batch, dim=0).to(device)
                res = model(batch).cpu().numpy()
                all_data.append(res)
                batch = []
        if batch:
            batch = torch.stack(batch, dim=0).to(device)
            res = model(batch).cpu().numpy()
            all_data.append(res)
    model.train()

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.stack(all_labels, axis=0)

    print(all_data.shape, all_labels.shape)
    pos = []
    neg = []
    for i in range(min(max_query_img, all_labels.size)):
        sim = all_data[i].reshape(1, -1) @ all_data.T
        mask_pos = all_labels[i] == all_labels
        mask_pos[i] = False
        mask_neg = all_labels[i] != all_labels
        pos.append(sim[mask_pos.reshape(1, -1)])
        neg.append(sim[mask_neg.reshape(1, -1)])

    pos = np.concatenate(pos, axis=0)
    neg = np.concatenate(neg, axis=0)
    labels = np.concatenate([np.zeros_like(neg), np.ones_like(pos)], axis=0)
    scores = np.concatenate([neg, pos], axis=0)
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    auc = roc_auc_score(labels, scores)

    fig, ax = plt.subplots()
    ax.semilogx(fpr, tpr, label=f'AUC = {auc}')
    ax.legend()
    plt.xlim([0.000001, 1])
    plt.savefig(f'cp-auc-{name}-{iteration:07d}.png')
    plt.close('all')

    return auc

