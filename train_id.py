import argparse
import sys
import time
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2
import torchvision
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import logging
import lmdb
from sklearn.metrics import roc_curve, roc_auc_score
from collections import defaultdict
from nets import net_factory


class IdDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path, augment=True, size_multiplier=1):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.txn = None
        self.augment = augment
        self.aug = None
        self.size_multiplier = size_multiplier
        self.key_index = 1

        with lmdb.open(self.lmdb_path, readonly=True) as env:
            with env.begin(write=False) as txn:
                all_keys = list(txn.cursor().iternext(values=False))

        self.keys = defaultdict(list)
        for k in all_keys:
            self.keys[k.decode().split('_')[self.key_index]].append(k)

        #if not self.augment:
        #    self.restrict_data()

        self.key_list = list(self.keys)
        self.all_keys = all_keys
        self.key_to_idx = {k: i for i, k in enumerate(self.key_list)}

    def restrict_data(self, max_id=2000, max_id_size=2):
        import random
        ids = list(self.keys)
        random.shuffle(ids)
        ids = ids[:max_id]
        for i in self.keys:
            random.shuffle(self.keys[i])
        self.keys = {i: self.keys[i][:max_id_size] for i in ids}

    def __len__(self):
        return len(self.key_list) * self.size_multiplier

    def _read_img(self, name):
        data = self.txn.get(name)
        if data is None:
            print(
                f"Unable to load value for key '{name.decode()}' from DB '{self.lmdb_path}'.", file=sys.stderr)
            return None
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Unable to decode image '{name.decode()}'.", file=sys.stderr)
            return None

        return image

    def init(self):
        if self.txn is None:
            env = lmdb.open(self.lmdb_path, readonly=True)
            self.txn = env.begin(write=False)
            if self.augment:
                from augmentation import AUGMENTATIONS
                self.aug = AUGMENTATIONS[self.augment]


    def image_count(self):
        return len(self.key_list)

    def get_image(self, idx: int) -> (np.ndarray, int):
        self.init()
        key = self.all_keys[idx]
        image = self._read_img(key)
        return image, str(self.key_to_idx[key.decode().split('_')[self.key_index]])

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, int):
        self.init()

        idx = idx % len(self.key_list)
        key_idx = self.key_list[idx]
        if len(self.keys[key_idx]) == 1:
            keys = [self.keys[key_idx][0], self.keys[key_idx][0]]
        else:
            keys = np.random.choice(self.keys[key_idx], 2, replace=False)

        images = [self._read_img(key) for key in keys]
        if self.aug is not None:
            image1, image2 = self.aug(images=images)
        else:
            image1, image2 = images

        image1 = torch.from_numpy(image1).permute(2, 0, 1)
        image2 = torch.from_numpy(image2).permute(2, 0, 1)

        return image1, image2, idx


def parseargs():
    parser = argparse.ArgumentParser(usage='Trains contrastive self-supervised training on artificial data.')
    parser.add_argument('--lmdb', required=True, help='Path to lmdb DB..')
    parser.add_argument('--lmdb-tst', required=False, help='Path to lmdb DB..')

    parser.add_argument('--start-iteration', default=0, type=int)
    parser.add_argument('--max-iteration', default=500000, type=int)
    parser.add_argument('--view-step', default=50, type=int, help="Number of training iterations between network testing.")

    parser.add_argument('--emb-dim', default=128, type=int, help="Output embedding dimension.")
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--learning-rate', default=0.0002, type=float)
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--emb-reg-weight', default=1, type=float)
    parser.add_argument('--augmentation', default='LITE')
    parser.add_argument('--warmup-iterations', default=100, type=float)
    parser.add_argument('--train-only-decoder', action='store_true')

    parser.add_argument('--backbone-config', default='{"type":"SM","name":"resnet34","weights":"imagenet","depth":5}')
    parser.add_argument('--decoder-config', default='{"type":"avg_pool"}')

    parser.add_argument('--loader-count', default=6, type=int, help="Number of processes loading training data.")
    parser.add_argument('--batch-history', default=0, type=int, help="Number of old batches used for distance matrix.")


    args = parser.parse_args()
    return args


def my_loss(emb, labels, old_emb=None, old_labels=None):
    sim = emb @ emb.t()

    # mask the same images on the main diagonal
    sim.fill_diagonal_(-1e20)

    # maximum value for stable exp --- without the main diagonal which is not used
    with torch.no_grad():
        max_val = torch.amax(sim, dim=1, keepdim=True)
    sim = torch.exp(sim - max_val)
    if old_emb is not None:
        # mask old positives
        sim_old = emb @ old_emb.t()
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
    loss = -torch.log(numerator / denominator + 1e-20) * pos
    loss = torch.sum(loss) / torch.sum(pos)
    
    return loss


def test(iteration, name, model, dataset, device, max_img, max_query_img):
    batch = []
    all_data = []
    all_labels = []
    print(max_img, len(dataset))
    with torch.no_grad():
        model = model.eval()
        for i in range(min(max_img, len(dataset))):
            img1, img2, label = dataset[i]
            batch.append(img1)
            batch.append(img2)
            all_labels.append(label)
            all_labels.append(label)
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


def test_retrieval(iteration, name, model, dataset, device, max_img=2000, batch_size=32, query_vis_count=16, result_vis_count=32):
    all_images = []
    all_labels = []
    for i in range(min(max_img, dataset.image_count())):
        img, label = dataset.get_image(i)
        all_images.append(img)
        all_labels.append(label)

    embeddings = []
    img_to_process = list(all_images)
    model = model.eval()
    with torch.no_grad():
        while img_to_process:
            batch = img_to_process[:batch_size]
            img_to_process = img_to_process[batch_size:]
            batch = np.stack(batch, axis=0)
            batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
            res = model(batch)
            embeddings.append(res)
    model = model.train()

    embeddings = torch.cat(embeddings, dim=0)
    similarities = embeddings @ embeddings.t()
    similarities = similarities.cpu().numpy()
    all_labels = np.asarray(all_labels)

    logging.info(f'SIMILARITIES {similarities.min()} {similarities.max()}, {similarities.shape}')


    collage_ids = np.linspace(0, len(all_images) - 1, query_vis_count).astype(np.int)
    result_collage = []

    scores = []
    labels = []
    for i in range(similarities.shape[0]):
        batch_sim = similarities[i]
        batch_labels = all_labels[i] == all_labels
        # remove the query image
        batch_labels[i] = False
        batch_sim[i] = -1e20
        scores.append(batch_sim)
        labels.append(batch_labels)

        if i in collage_ids:
            row = [all_images[i]]
            best_ids = np.argsort(batch_sim)[::-1][:result_vis_count]
            for id in best_ids:
                row.append(all_images[id])
            result_collage.append(np.concatenate(row, axis=1))

    if result_collage:
        result_collage = np.concatenate(result_collage, axis=0)
        cv2.imwrite(f'result-{name}-{iteration:07d}.jpg', result_collage)

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)

    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    auc = roc_auc_score(labels, scores)
    return auc, fpr, tpr, thr


def tile_images(images, step=16):
    lines = [np.concatenate(images[i:i+step], axis=1) for i in range(0, len(images), step)]
    return np.concatenate(lines, axis=0)


def main():
    args = parseargs()
    logging.info(f'ARGS {args}')

    device = torch.device('cuda')
    #torch.multiprocessing.set_start_method('spawn')

    img_encoder = net_factory(args.backbone_config, args.decoder_config, emb_dim=args.emb_dim).to(device)

    if args.start_iteration > 0:
        checkpoint_path = f'cp-{args.start_iteration:07d}.img.ckpt'
        logging.info(f'Loading image checkpoint {checkpoint_path}')
        img_encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))

    img_encoder.eval()
    #input_names = ["input_image"]
    #output_names = ["output_emb"]
    #torch.onnx.export(img_encoder, torch.from_numpy(np.zeros((1, 3, 64, 64), dtype=np.float32)).to(device), 'model.onnx', verbose=False, input_names=input_names, output_names=output_names, opset_version=11)

    ds_trn = IdDataset(args.lmdb, augment=args.augmentation)
    dl_trn = DataLoader(ds_trn, num_workers=args.loader_count, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        persistent_workers=True, pin_memory=True, prefetch_factor=2)
    print(f'TRN DATASET LEN: {len(ds_trn)}')

    if args.lmdb_tst:
        ds_tst = IdDataset(args.lmdb_tst, augment=None, size_multiplier=1)
        print(f'TST DATASET LEN: {len(ds_tst)}')
    else:
        ds_tst = None

    if args.train_only_decoder:
        optimizer = torch.optim.AdamW(img_encoder.decoder.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(img_encoder.parameters(), lr=args.learning_rate)

    loss_history = []
    iteration = args.start_iteration

    to_test = 20
    test_similarities = []
    test_labels = []
    label_history = []
    embed_history = []

    t1 = time.time()
    for epoch in range(10000):
        for images1, images2, labels in dl_trn:
            if iteration == args.start_iteration:
                images = []
                for i1, i2 in zip(images1.permute(0, 2, 3, 1).numpy(), images2.permute(0, 2, 3, 1).numpy()):
                    images.append(i1)
                    images.append(i2)
                cv2.imwrite('images.jpg', tile_images(images), params=[int(cv2.IMWRITE_JPEG_QUALITY), 98])


            #print(unique_labels, ' '.join([str(i) for i in labels]))
            #unique_labels = len(set([i.item() for i in labels]))
            #if unique_labels != images1.shape[0]:
            #    print(f'Unique labels in {iteration}: {unique_labels}')
            with torch.no_grad():
                images1 = images1.to(device)
                images2 = images2.to(device)
                images = torch.cat([images1, images2], dim=0)
                labels = torch.cat([labels, labels]).to(device)

            if iteration - args.start_iteration <= args.warmup_iterations:
                lr = (min(iteration - args.start_iteration, args.warmup_iterations) / args.warmup_iterations) ** 1.5 * args.learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            optimizer.zero_grad()
            embedding = img_encoder(images)
            #if label_history:
            #    all_emb = torch.cat([e for e in embed_history] + [embedding], dim=0)
            #    all_labels = torch.cat([e for e in label_history] + [labels], dim=0)
            #else:
            #    all_emb = embedding
            #    all_labels = labels

            #    #miner_output = miner(all_emb, all_labels)
            #    #loss = loss_object(all_emb, all_labels, miner_output)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if label_history:
                    loss = my_loss(embedding, labels, torch.cat(embed_history, dim=0), torch.cat(label_history, dim=0))
                else:
                    loss = my_loss(embedding, labels)

            loss = loss + args.emb_reg_weight * torch.mean(embedding ** 2)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

            #sim = torch.mm(embedding, embedding.t()).detach().cpu().numpy()
            #print(f'SIM: {lr} {iteration} {sim.min()} {sim.max()}')

            if args.batch_history:
                with torch.no_grad():
                    embed_history.append(embedding.detach())
                    label_history.append(labels.detach())
                label_history = label_history[:args.batch_history]
                embed_history = embed_history[:args.batch_history]

            if iteration % args.view_step > args.view_step - to_test:
                test_labels.append(labels.cpu().numpy())
                test_similarities.append(torch.mm(embedding, embedding.t()).detach().cpu().numpy())

            if iteration % args.view_step == 0 and iteration > args.start_iteration:
                print('SIMILARITIES', test_similarities[-1].min(), test_similarities[-1].max())
                fig, ax = plt.subplots()
                heatmap = ax.imshow(test_similarities[-1])
                plt.colorbar(heatmap)
                plt.savefig(f'cp-{iteration:07d}.png')
                plt.close('all')

                test_results = {}
                if ds_tst is not None:
                    tst_auc, fpr, tpr, thr = test_retrieval(iteration, 'tst', img_encoder, ds_tst, device, max_img=2000, batch_size=32)
                    test_results['tst'] = (tst_auc, fpr, tpr, thr)
                else:
                    tst_auc = -1
                trn_auc, fpr, tpr, thr = test_retrieval(iteration, 'trn', img_encoder, ds_trn, device, max_img=2000, batch_size=32)
                test_results['trn'] = (trn_auc, fpr, tpr, thr)

                avg_loss = np.mean(loss_history)
                torch.save(img_encoder.state_dict(), f'cp-{iteration:07d}.img.ckpt')
                print(f'LOG {iteration} iterations:{iteration} trn_auc:{trn_auc:0.6f} tst_auc:{tst_auc:0.6f} loss:{avg_loss:0.6f} time:{time.time()-t1:.1f}s')
                t1 = time.time()
                test_similarities = []
                test_labels = []
                loss_history = []

                fig, ax = plt.subplots()
                for name, (auc, fpr, tpr, thr) in test_results.items():
                    ax.semilogx(fpr, tpr, label=f'{name} AUC = {auc}')
                ax.legend()
                plt.xlim([0.000001, 1])
                plt.savefig(f'cp-auc-{iteration:07d}.png')
                plt.close('all')

                # if dl_tst is not None:
                #    #test(f'tsne-{iteration:07d}.trn.png', img_encoder, dl_trn, device)
                #    test(f'tsne-{iteration:07d}.tst.png', img_encoder, dl_tst, device)
            iteration += 1


if __name__ == '__main__':
    main()
