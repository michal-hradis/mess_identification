import random
import sys
import torch
import lmdb
import cv2
from collections import defaultdict

import numpy as np

from code.augmentation import AUGMENTATIONS


class IdDataset(torch.utils.data.Dataset):
    """
    Dataset for identity-based sampling from an LMDB database.

    Overview
    - Samples images grouped by identity (class) from an LMDB where each database key string encodes
      image class and video UUID joined by underscores.
    - Returns either a single image or a pair of images from the same class
      (useful for contrastive/siamese training).

    Constructor parameters
    - lmdb_path (str): path to the LMDB database.
    - augment (str|None): key into AUGMENTATIONS to apply on sampled images.
    - size_multiplier (int): legacy parameter kept for compatibility (not used in sampling).
    - single_image (bool): if True, __getitem__ returns a single image sample, otherwise a pair of images of the same class.
    - key_index (list[int]): indices use to select a class ID from the LMDB key string.

    Key parsing
    - Keys are decoded and split on '_' and the parts indicated by `key_index` are
      joined to form the image id; the first selected part is treated as the video UUID.

    Indexing and length
    - __len__() returns the number of distinct classes (not the total number of images).
    - __getitem__(idx) treats idx as a class index and samples images belonging to that class.

    Returned tensors and fields
    - Images are returned as torch.Tensor with shape (C, H, W), RGB channel order, and dtype torch.uint8.
    - Single-image return example: {'image': tensor, 'label': int, 'video_id': int}
    - Paired-image return example: {'image1': tensor, 'image2': tensor, 'label': int, 'video_id': int}

    Important methods
    - init(): lazily opens the LMDB transaction and loads augmentation callable.
    - _read_img(name): decodes LMDB value with cv2.imdecode and returns a numpy image.
    - get_image(image_id): returns (image_numpy, class_int, video_int) for a given key index.
    - get_all_id_images(class_idx): returns all images belonging to a class as numpy arrays.
    """
    def __init__(self, lmdb_path, augment=None, size_multiplier=1, single_image=False, key_index=None):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.txn = None
        self.augment = augment
        self.aug = None
        self.size_multiplier = size_multiplier
        # avoid mutable default argument
        if key_index is None:
            key_index = [0, 1]
        self.label_key_index = key_index
        self.single_image = single_image

        with lmdb.open(self.lmdb_path, readonly=True, readahead=False) as env:
            with env.begin(write=False) as txn:
                keys = list(txn.cursor().iternext(values=False))

        self.keys = keys
        self.keys_parsed = [self.parse_key(k) for k in keys]
        all_classes = set((i[0] for i in self.keys_parsed))
        all_video_uuids = set((i[1] for i in self.keys_parsed))
        self.class_to_int = {class_name: idx for idx, class_name in enumerate(sorted(all_classes))}
        self.video_uuid_to_int = {video_uuid: idx for idx, video_uuid in enumerate(sorted(all_video_uuids))}
        self.keys_parsed_int = [(self.class_to_int[class_name], self.video_uuid_to_int[video_uuid])
                                for class_name, video_uuid in self.keys_parsed]

        self.class_sample_list = defaultdict(list)
        for i, (class_id, video_id) in enumerate(self.keys_parsed_int):
            self.class_sample_list[class_id].append(i)

    def parse_key(self, k: bytes) -> tuple[str, str]:
        """
        Returns:
            Image ID, video UUID
        """
        k = k.decode().split('_')
        k = [k[i] for i in self.label_key_index]
        return '_'.join(k), k[0]

    def __len__(self):
        return len(self.class_to_int)

    def _read_img(self, name):
        data = self.txn.get(name)
        if data is None:
            print(
                f"Unable to load value for key '{name.decode()}' from DB '{self.lmdb_path}'.", file=sys.stderr)
            return None
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)[:,:, ::-1]
        if image is None:
            print(f"Unable to decode image '{name.decode()}'.", file=sys.stderr)
            return None

        return image

    def init(self):
        if self.txn is None:
            env = lmdb.open(self.lmdb_path, readonly=True, readahead=False)
            self.txn = env.begin(write=False)
            if self.augment:
                self.aug = AUGMENTATIONS[self.augment]

    def image_count(self):
        return len(self.keys)

    def get_image(self, image_id: int) -> tuple[np.ndarray, int, int]:
        self.init()
        key = self.keys[image_id]
        image = self._read_img(key)
        class_str, video_str = self.parse_key(key)
        return image, self.class_to_int[class_str], self.video_uuid_to_int[video_str]

    def get_all_id_images(self, class_idx: int) -> list[np.ndarray]:
        self.init()
        class_samples = self.class_sample_list[class_idx]
        images = [self._read_img(self.keys[image_id]) for image_id in class_samples]
        return images

    def __getitem__(self, class_idx):
        self.init()

        class_idx = class_idx % len(self.class_sample_list)
        class_samples = self.class_sample_list[class_idx]
        class_id, video_id = self.keys_parsed_int[class_samples[0]]

        if self.single_image:
            selected_sample = random.choice(class_samples)
            image, image_class_id, image_video_id = self.get_image(selected_sample)
            images = [image]
            if self.aug is not None:
                image, = self.aug(images=images)
            else:
                image, = images
            return {
                'image': torch.from_numpy(image).permute(2, 0, 1),
                'label': class_id,
                'video_id': video_id
            }
        elif len(class_samples) == 1:
            selected_samples = [class_samples[0], class_samples[0]]
        else:
            selected_samples = np.random.choice(class_samples, 2, replace=False)

        images = [self.get_image(sample_id) for sample_id in selected_samples]
        if self.aug is not None:
            image1, image2 = self.aug(images=images)
        else:
            image1, image2 = images

        image1 = torch.from_numpy(image1).permute(2, 0, 1)
        image2 = torch.from_numpy(image2).permute(2, 0, 1)

        return {
            'image1': image1,
            'image2': image2,
            'label': class_idx,
            'video_id': video_id
        }
