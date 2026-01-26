import argparse
import time
import logging

from torch.utils.data import DataLoader
import torch
import cv2

from code.id_dataset import IdDataset
from code.nets import net_factory
from code.augmentation import AUGMENTATIONS
from code.domain_adaptation import EmbeddingModelWithDomainAdaptation
from code.losses import get_loss_function
from code.evaluation import RetrievalEvaluator
from trainer import IdentityTrainer

LOSSES = ['normalized_softmax', 'xent', 'arcface']

def parseargs():
    parser = argparse.ArgumentParser(usage='Trains contrastive self-supervised networks on groups of images with augmentation '
                                           'The trained models should be used for retrieval, clustering, and fine-tuned for '
                                           'downstream tasks.')
    parser.add_argument('--lmdb', required=True, help='Path to lmdb DB..')
    parser.add_argument('--lmdb-tst', required=False, help='Path to lmdb DB..')
    parser.add_argument('--name', default='model', help='Name of the experiment and model.')
    parser.add_argument('--key-index', default = [0, 1], type=int, nargs='+', help='File names in lmdb (keys) contain ID. '
                                                                                   'The file name is split by "_" and the '
                                                                                   'indices are used to extract the unique identity ID.')
    parser.add_argument('--start-iteration', default=0, type=int)
    parser.add_argument('--max-iteration', default=500000, type=int)
    parser.add_argument('--view-step', default=50, type=int, help="Number of training iterations between network testing.")

    parser.add_argument('--emb-dim', default=128, type=int, help="Output embedding dimension.")
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--learning-rate', default=0.0002, type=float)
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--loss-learning-rate', default=0.01, type=float)
    parser.add_argument('--loss-weight-decay', default=0.0001, type=float)

    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--emb-reg-weight', default=1, type=float)
    parser.add_argument('--augmentation', default='LITE', help=f'Augmentation type. The options are {AUGMENTATIONS.keys()}.')
    parser.add_argument('--gpu-augmentation', help=f'Augmentation on GPU. The options are ["aug_1"].')
    parser.add_argument('--warmup-iterations', default=100, type=float)
    parser.add_argument('--train-only-decoder', action='store_true', help='The encoder is frozen and only the decoder is trained.')
    parser.add_argument('--no-emb-normalization', action='store_true', help="Do not normalize embeddings. Normalizes to unit length by default.")

    parser.add_argument('--backbone-config', default='{"type":"SM","name":"resnet34","weights":"imagenet","depth":5}')
    parser.add_argument('--decoder-config', default='{"type": "pool", "operation": "avg"}')

    parser.add_argument('--loader-count', default=6, type=int, help="Number of processes loading training data.")
    parser.add_argument('--batch-history', default=0, type=int, help="Number of old batches used for distance matrix.")

    parser.add_argument('--loss', default='xent', help=f'Loss function. The options are {LOSSES}.')

    # Domain Adaptation arguments
    parser.add_argument('--use-domain-adaptation', action='store_true',
                        help='Enable domain adaptation with gradient reversal to improve domain robustness.')
    parser.add_argument('--domain-loss-weight', default=0.1, type=float,
                        help='Weight for the domain adaptation loss.')
    parser.add_argument('--domain-hidden-dims', default=[256, 128], type=int, nargs='+',
                        help='Hidden layer dimensions for domain classifier MLP.')
    parser.add_argument('--domain-dropout', default=0.0, type=float,
                        help='Dropout probability for domain classifier.')
    parser.add_argument('--domain-grl-lambda', default=1.0, type=float,
                        help='Gradient reversal lambda parameter (strength of gradient reversal).')

    args = parser.parse_args()
    return args




def main():
    args = parseargs()
    logging.basicConfig(level=logging.INFO)
    logging.info(f'ARGS {args}')

    device = torch.device('cuda')

    # Create base model
    base_model = net_factory(
        args.backbone_config,
        args.decoder_config,
        emb_dim=args.emb_dim,
        normalize=not args.no_emb_normalization
    ).to(device)

    print(base_model)

    # Load checkpoint if resuming training
    if args.start_iteration > 0:
        checkpoint_path = f'cp-{args.start_iteration:07d}.img.ckpt'
        logging.info(f'Loading image checkpoint {checkpoint_path}')
        base_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logging.info(f'LOADED image checkpoint {checkpoint_path}')

    # Setup datasets
    single_image = args.loss in ['normalized_softmax']
    size_multiplier = 1

    ds_trn = IdDataset(
        args.lmdb,
        augment=args.augmentation,
        single_image=single_image,
        size_multiplier=size_multiplier,
        key_index=args.key_index
    )
    dl_trn = DataLoader(
        ds_trn,
        num_workers=args.loader_count,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True
    )
    logging.info(f'TRN DATASET LEN: {len(ds_trn)}')

    if args.lmdb_tst:
        ds_tst = IdDataset(
            args.lmdb_tst,
            augment=None,
            size_multiplier=1,
            single_image=single_image,
            key_index=args.key_index
        )
        logging.info(f'TST DATASET LEN: {len(ds_tst)}')
    else:
        ds_tst = None

    # Wrap model with domain adaptation if enabled
    if args.use_domain_adaptation:
        num_domains = len(ds_trn.video_uuid_to_int)
        logging.info(f'Domain Adaptation enabled with {num_domains} domains (video_ids)')
        logging.info(f'Domain classifier hidden dims: {args.domain_hidden_dims}')
        logging.info(f'Domain classifier classes: {num_domains}')
        logging.info(f'Domain loss weight: {args.domain_loss_weight}')
        logging.info(f'Domain GRL lambda: {args.domain_grl_lambda}')

        model = EmbeddingModelWithDomainAdaptation(
            base_model,
            num_domains=num_domains,
            domain_hidden_dims=args.domain_hidden_dims,
            domain_dropout=args.domain_dropout
        ).to(device)
        model.set_grl_lambda(args.domain_grl_lambda)
    else:
        model = base_model
        logging.info('Domain Adaptation disabled')

    # Create loss function and optimizer
    loss_fce, loss_optimizer = get_loss_function(args.loss, args, len(ds_trn), device)

    # Create model optimizer
    if args.train_only_decoder:
        optimizer = torch.optim.AdamW(
            model.decoder.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    # Create trainer and evaluator
    trainer = IdentityTrainer(model, loss_fce, optimizer, args, device, loss_optimizer)
    evaluator = RetrievalEvaluator(device, batch_size=32)

    t1 = time.time()

    # Training loop
    for epoch in range(10000):
        if trainer.is_max_iteration_reached():
            break

        for batch_data in dl_trn:
            if trainer.is_max_iteration_reached():
                break

            # Save first batch for visualization
            if trainer.iteration == args.start_iteration:
                print(batch_data['label'])
                # Could save training batch visualization here if needed
                # images = prepare_batch_visualization(batch_data)
                # cv2.imwrite('images.jpg', tile_images(images), params=[int(cv2.IMWRITE_JPEG_QUALITY), 98])

            # Perform training step
            loss, embedding, images = trainer.train_step(batch_data)
            labels = batch_data['label'].to(device)

            # Track test data if near evaluation point
            if trainer.is_test_tracking():
                trainer.track_test_data(embedding, labels)

            # Evaluation and checkpointing
            if trainer.should_evaluate():
                # Export model
                trainer.export_model(images)

                # Plot similarity heatmap
                trainer.plot_similarity_heatmap()

                # Evaluate on test and train sets
                max_test_img = 5000
                test_results = {}

                if ds_tst is not None:
                    logging.info('Evaluating on test set...')
                    tst_results = evaluator.evaluate_retrieval(
                        model, ds_tst, max_img=max_test_img
                    )
                    test_results['tst'] = tst_results

                    # Save test result collage
                    tst_collage = evaluator.create_result_collage(
                        tst_results['all_images'],
                        tst_results['all_labels'],
                        tst_results['similarities']
                    )
                    if tst_collage is not None:
                        cv2.imwrite(f'result-tst-{trainer.iteration:07d}.jpg', tst_collage[:, :, ::-1])
                else:
                    tst_results = None

                logging.info('Evaluating on training set...')
                trn_results = evaluator.evaluate_retrieval(
                    model, ds_trn, max_img=max_test_img
                )
                test_results['trn'] = trn_results

                # Save train result collage
                trn_collage = evaluator.create_result_collage(
                    trn_results['all_images'],
                    trn_results['all_labels'],
                    trn_results['similarities']
                )
                if trn_collage is not None:
                    # `trn_collage` is in RGB (internal representation). Convert to BGR for OpenCV imwrite.
                    cv2.imwrite(f'result-trn-{trainer.iteration:07d}.jpg', trn_collage[:, :, ::-1])

                # Save checkpoint
                trainer.save_checkpoint()

                # Log results
                losses_str = trainer.get_loss_summary()
                trn_auc = trn_results['auc']
                trn_mean_auc = trn_results['mean_auc']
                trn_map = trn_results['mean_ap']

                if tst_results is not None:
                    tst_auc = tst_results['auc']
                    tst_mean_auc = tst_results['mean_auc']
                    tst_map = tst_results['mean_ap']
                else:
                    tst_auc = -1
                    tst_mean_auc = -1
                    tst_map = -1

                print(f'LOG {trainer.iteration} iterations:{trainer.iteration} {losses_str} '
                      f'trn_auc:{trn_auc:0.6f} trn_mauc:{trn_mean_auc:0.6f} trn_map:{trn_map:0.6f} '
                      f'tst_auc:{tst_auc:0.6f} tst_mauc:{tst_mean_auc:0.6f} tst_map:{tst_map:0.6f} '
                      f'time:{time.time()-t1:.1f}s')

                # Plot ROC curves
                evaluator.plot_roc_curve(test_results, f'cp-auc-{trainer.iteration:07d}.png')

                # Reset tracking
                t1 = time.time()
                trainer.reset_test_tracking()

            trainer.increment_iteration()


if __name__ == '__main__':
    main()
