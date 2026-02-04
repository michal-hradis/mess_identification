import argparse

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
import lmdb
import torchvision
from train_id import Encoder2


def parseargs():
    parser = argparse.ArgumentParser(usage='Trains contrastive self-supervised training on artificial data.')
    parser.add_argument('--lmdb', required=True, help='Path to lmdb DB..')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--max-img', default=5000, type=int, help='Maximum number of processed images.')
    parser.add_argument('--emb-dim', default=128, type=int, help="Output embedding dimension.")
    args = parser.parse_args()
    return args


def show_results(txn, ids, image_keys):
    print('a', len(image_keys))
    print('b', ids)
    image_keys = [image_keys[i] for i in ids]
    data = [txn.get(image_key) for image_key in image_keys]
    # decode (BGR) -> convert to RGB for internal representation
    images = [cv2.imdecode(np.frombuffer(d, dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1] for d in data]
    lines = [np.concatenate(images[i:i+8], axis=1) for i in range(0, len(images), 8)]
    return np.concatenate(lines, axis=0)


def main():
    args = parseargs()
    logging.info(f'ARGS {args}')

    device = torch.device('cuda')
    env = lmdb.open(args.lmdb, readonly=True)
    txn = env.begin(write=False)

    img_encoder = Encoder2(args.emb_dim, 32).to(device)
    img_encoder.load_state_dict(torch.load(args.checkpoint, map_location=device))
    img_encoder = img_encoder.eval()
    all_keys = list(txn.cursor().iternext(values=False))[:args.max_img]
    print(all_keys)

    embeddings = []
    transform = torchvision.transforms.ToTensor()
    with torch.no_grad():
        for i, image_key in enumerate(all_keys):
            data = txn.get(image_key)
            # decode BGR -> convert to RGB
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
            image = torch.unsqueeze(transform(image), dim=0).to(device)
            embeddings.append(img_encoder(image).cpu().numpy())
            if i % 100 == 0:
                print('DONE', i)

    embeddings = np.concatenate(embeddings, axis=0)
    print(embeddings.shape)

    while True:
        id = np.random.randint(embeddings.shape[0])
        sim = embeddings @ embeddings[id][:, np.newaxis]
        print(sim.shape)
        sim = sim[:, 0]
        most_similar = [id] + np.argsort(sim).tolist()[::-1]
        print(sim[most_similar[:64]], sim.min())
        images = show_results(txn, most_similar[:64], all_keys)
        # show_results returns an RGB collage; convert to BGR for OpenCV display
        cv2.imshow('results', images[:, :, ::-1])
        key = cv2.waitKey()
        if key == 27:
            break


if __name__ == '__main__':
    main()
