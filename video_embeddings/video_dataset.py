"""Video dataset for temporal embedding learning."""
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import lmdb
import cv2


class VideoDataset(torch.utils.data.Dataset):
    """
    Dataset for sampling frames from videos stored in LMDB or directory.

    File naming convention: {video_id}_{frame_id}.jpg

    Args:
        data_path: Path to directory or LMDB containing video frames
        num_frames: Number of frames to sample from each video
        max_frame_gap: Maximum gap between consecutive frames
        image_size: Tuple of (height, width) to resize images to
        is_lmdb: Whether data_path is an LMDB database
    """

    def __init__(
        self,
        data_path: str,
        num_frames: int = 8,
        max_frame_gap: int = 5,
        image_size: tuple[int, int] = (224, 224),
        is_lmdb: bool = True
    ):
        super().__init__()
        self.data_path = data_path
        self.num_frames = num_frames
        self.max_frame_gap = max_frame_gap
        self.image_size = image_size
        self.is_lmdb = is_lmdb

        # Lazy initialization
        self.txn = None

        # Index videos and frames
        if is_lmdb:
            self._index_lmdb()
        else:
            self._index_directory()

    def _index_lmdb(self):
        """Index frames in LMDB database grouped by video_id."""
        with lmdb.open(self.data_path, readonly=True, readahead=False) as env:
            with env.begin(write=False) as txn:
                keys = [k.decode() for k in txn.cursor().iternext(values=False)]

        self._index_from_keys(keys)

    def _index_directory(self):
        """Index frames in directory grouped by video_id."""
        path = Path(self.data_path)
        keys = [f.stem for f in path.glob("*.jpg")]
        self._index_from_keys(keys)

    def _index_from_keys(self, keys: list[str]):
        """Build video_id -> frames mapping from keys."""
        self.video_frames = defaultdict(list)

        for key in keys:
            parts = key.split('_')
            if len(parts) < 2:
                continue

            video_id = parts[0]
            try:
                frame_id = int(parts[1])
                self.video_frames[video_id].append((frame_id, key))
            except (ValueError, IndexError):
                continue

        # Sort frames by frame_id for each video
        for video_id in self.video_frames:
            self.video_frames[video_id].sort(key=lambda x: x[0])

        # Create list of video_ids with sufficient frames
        self.video_ids = [
            vid for vid, frames in self.video_frames.items()
            if len(frames) >= self.num_frames
        ]

        if len(self.video_ids) == 0:
            raise ValueError(f"No videos found with at least {self.num_frames} frames")

    def _init_lmdb(self):
        """Lazily initialize LMDB connection."""
        if self.is_lmdb and self.txn is None:
            env = lmdb.open(self.data_path, readonly=True, readahead=False)
            self.txn = env.begin(write=False)

    def _read_frame(self, key: str) -> np.ndarray:
        """Read and decode a single frame."""
        if self.is_lmdb:
            self._init_lmdb()
            data = self.txn.get(key.encode())
            if data is None:
                raise ValueError(f"Frame {key} not found in LMDB")
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            image_path = Path(self.data_path) / f"{key}.jpg"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to decode frame {key}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

        return image

    def _sample_frame_indices(self, num_available: int) -> list[int]:
        """Sample consecutive frame indices with random gaps."""
        # Calculate maximum starting index
        max_span = (self.num_frames - 1) * self.max_frame_gap

        if num_available <= self.num_frames:
            # Not enough frames, sample with replacement
            return sorted(random.choices(range(num_available), k=self.num_frames))

        if num_available <= max_span:
            # Sample without maximum gap constraint
            start_idx = 0
            available_range = num_available - 1
        else:
            # Sample with gap constraint
            start_idx = random.randint(0, num_available - max_span - 1)
            available_range = max_span

        # Sample frame offsets
        frame_indices = [start_idx]
        current_idx = start_idx

        for _ in range(self.num_frames - 1):
            # Sample gap between 1 and max_frame_gap
            max_gap = min(self.max_frame_gap, start_idx + available_range - current_idx)
            gap = random.randint(1, max(1, max_gap))
            current_idx = min(current_idx + gap, start_idx + available_range)
            frame_indices.append(current_idx)

        return frame_indices

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - frames: uint8 tensor of shape (num_frames, 3, H, W)
                - video_id: integer video identifier
        """
        video_id = self.video_ids[idx]
        frames_list = self.video_frames[video_id]

        # Sample frame indices
        sampled_indices = self._sample_frame_indices(len(frames_list))

        # Load frames
        frames = []
        for idx in sampled_indices:
            _, key = frames_list[idx]
            frame = self._read_frame(key)
            frames.append(frame)

        # Stack to (num_frames, H, W, 3)
        frames = np.stack(frames, axis=0)

        # Convert to torch tensor (num_frames, 3, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

        return {
            'frames': frames,
            'video_id': idx
        }
