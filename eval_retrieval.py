import argparse

import torch
import numpy as np
import cv2
import logging
import lmdb
import torchvision
from sklearn.metrics import average_precision_score
from nets import net_factory


def parseargs():
    parser = argparse.ArgumentParser(usage='Evaluate retrieval quality as defined for VERI-Wild 2.0 dataset.')
    parser.add_argument('--lmdb', required=True, help='Path to lmdb DB..')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--query-list', required=True)
    parser.add_argument('--db-list', required=True)
    parser.add_argument('--emb-dim', default=128, type=int, help="Output embedding dimension.")
    parser.add_argument('--backbone-config', default='{"type":"SM","name":"resnet34","weights":"imagenet","depth":5}')
    parser.add_argument('--decoder-config', default='{"type": "pool", "operation": "avg"}')
    args = parser.parse_args()
    return args


def compute_embeddings(encoder, device, file, txn):
    with open(file, 'r') as f:
        images = [line.split() for line in f]

    encoder = encoder.eval()
    embeddings = []
    car_ids = []
    cam_ids = []
    transform = torchvision.transforms.ToTensor()
    count = 0
    batch = []
    batch_size = 32
    with torch.no_grad():
        for file_id, idx, cam_id in images:
            file_id = file_id.split('/')[1]
            car_ids.append(int(idx))
            cam_ids.append(int(cam_id))
            image_key = f"VehicleID-{idx}_{file_id}".encode()
            data = txn.get(image_key)
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            image = torch.unsqueeze(transform(image), dim=0)
            batch.append(image)
            if len(batch) > batch_size:
                batch = torch.cat(batch, dim=0).to(device)
                embeddings.append(encoder(batch).cpu().numpy())
                batch = []
            count += 1
            if count % 100 == 0:
                print(f'DONE {count}/{len(images)}')

        if len(batch) > 0:
            batch = torch.cat(batch, dim=0).to(device)
            embeddings.append(encoder(batch).cpu().numpy())

    car_ids = np.stack(car_ids)
    cam_ids = np.stack(cam_ids)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, car_ids, cam_ids


def main():
    args = parseargs()
    logging.info(f'ARGS {args}')

    device = torch.device('cuda')
    env = lmdb.open(args.lmdb, readonly=True)
    txn = env.begin(write=False)

    img_encoder = net_factory(args.backbone_config, args.decoder_config, emb_dim=args.emb_dim).to(device)
    img_encoder.load_state_dict(torch.load(args.checkpoint, map_location=device))
    img_encoder = img_encoder.eval()

    db_embeddings, db_car_ids, db_cam_ids = compute_embeddings(img_encoder, device, args.db_list, txn)
    q_embeddings, q_car_ids, q_cam_ids = compute_embeddings(img_encoder, device, args.query_list, txn)

    all_top_1 = []
    all_top_5 = []
    all_ap = []
    count = 0
    for q_emb, q_car_id, q_cam_id in zip(q_embeddings, q_car_ids, q_cam_ids):
        scores = db_embeddings @ q_emb.reshape(-1, 1)
        scores = scores.reshape(-1)
        relevant = np.logical_or(db_car_ids != q_car_id, db_cam_ids != q_cam_id)
        scores = scores[relevant]
        car_ids = db_car_ids[relevant] == q_car_id
        if not np.any(car_ids):
            print('ERROR: No relevant db car with the same ID:', q_car_id, q_cam_id)
            continue
        sort_i = np.argsort(scores)[::-1].reshape(-1)
        top_1 = car_ids[sort_i[0]]
        top_5 = np.any(car_ids[sort_i[:5]])
        ap = average_precision_score(car_ids, scores)
        all_top_1.append(top_1)
        all_top_5.append(top_5)
        all_ap.append(ap)
        #print(top_1, top_5, ap)
        count += 1
        if count % 100 == 0:
            print(f'DONE {count}/{q_car_ids.size}')

    print(f'mAP:{np.average(all_ap)} TOP_1:{np.average(all_top_1)} TOP_5:{np.average(all_top_5)}')

if __name__ == '__main__':
    main()
