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

    def _sample_frame_indices(self, frames_list: list[tuple[int, str]]) -> list[int]:
        """Sample consecutive frame indices respecting actual frame_id gaps.

        Args:
            frames_list: List of (frame_id, key) tuples sorted by frame_id

        Returns:
            List of indices into frames_list representing a valid sequence
        """
        num_available = len(frames_list)

        if num_available <= self.num_frames:
            # Not enough frames, sample with replacement
            return sorted(random.choices(range(num_available), k=self.num_frames))

        # Find all valid starting positions
        valid_starts = []

        for start_idx in range(num_available - self.num_frames + 1):
            # Check if we can get num_frames starting from start_idx
            # with all gaps <= max_frame_gap
            is_valid = True

            for i in range(self.num_frames - 1):
                frame_id_curr = frames_list[start_idx + i][0]
                frame_id_next = frames_list[start_idx + i + 1][0]
                gap = frame_id_next - frame_id_curr

                if gap > self.max_frame_gap:
                    is_valid = False
                    break

            if is_valid:
                valid_starts.append(start_idx)

        if not valid_starts:
            # No valid sequence found, fall back to sampling any consecutive frames
            # This can happen if max_frame_gap is too restrictive
            start_idx = random.randint(0, num_available - self.num_frames)
            return list(range(start_idx, start_idx + self.num_frames))

        # Randomly select one valid starting position
        start_idx = random.choice(valid_starts)
        return list(range(start_idx, start_idx + self.num_frames))

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
        sampled_indices = self._sample_frame_indices(frames_list)

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
