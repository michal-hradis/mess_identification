import torch
import numpy as np
import cv2
import json
import argparse
from torchcodec.decoders import VideoDecoder
from code.nets import net_factory
import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Load video, extract frames, extract features, and images.')
    parser.add_argument('--video-file', type=str, required=True, help='Path to the video file.')
    parser.add_argument('--score-file', type=str, required=True, help='Path to the score file containing visibility scores (json).')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--frame-rate', type=float, default=1, help='Frame rate to extract frames at in fps.')
    parser.add_argument('--image-resolution', type=int, default=320, help='Resolution to resize images to.')
    parser.add_argument('--net-resolution', type=int, default=128, help='Resolution to resize frames to.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--backbone-config', type=str, required=True, help='Path to the backbone config file.')
    parser.add_argument('--decoder-config', type=str, required=True, help='Path to the decoder config file.')
    parser.add_argument('--emb-dim', type=int, default=256, help='Embedding dimension.')
    parser.add_argument('--no-emb-normalization', action='store_true', help='Do not normalize embeddings.')
    return parser.parse_args()


def binary_search_lower(arr, low, high, x):


    mid = (low + high) // 2
    if high - low <= 1:
        return low
    if arr[mid][0] < x:
        return binary_search_lower(arr, mid, high, x)
    else:
        return binary_search_lower(arr, low, mid, x)


def find_closest_index(arr, x):
    low_element = binary_search_lower(arr, 0, len(arr), x)
    left_distance = abs(arr[low_element][0] - x)
    right_distance = abs(arr[low_element + 1][0] - x) if low_element + 1 < len(arr) else 1e20
    if left_distance < right_distance:
        return low_element
    else:
        return low_element + 1

def find_score(scores: list[tuple[float, float]], t: float) -> float:
    idx = find_closest_index(scores, t)
    return scores[idx][1]


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net_factory(args.backbone_config, args.decoder_config, emb_dim=args.emb_dim,
                        normalize=not args.no_emb_normalization
                        ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.eval()

    with open(args.score_file, 'r') as f:
        scores = json.load(f)[0]["data"]
    scores = json.loads(scores)
    # scores are in the format of [frame_time, score]

    video_last_timestamp = scores[-1][0]

    video_decoder = VideoDecoder(args.video_file, device="cpu", num_ffmpeg_threads=6)


    frame_times = []
    frames_features = []
    frame_id = 0
    for t in tqdm.tqdm(np.arange(0, video_last_timestamp, 1 / args.frame_rate)):
        visibility_score = find_score(scores, t)
        if visibility_score < 0.5:
            continue
        frame = video_decoder.get_frame_played_at(t)
        frame = frame.data.numpy().transpose(1, 2, 0)
        frame_image = cv2.resize(frame, (args.image_resolution, args.image_resolution), interpolation=cv2.INTER_AREA)

        frame_net = cv2.resize(frame, (args.net_resolution, args.net_resolution), interpolation=cv2.INTER_AREA)
        frame_net = frame_net[:, :, ::-1].transpose(2, 0, 1)[np.newaxis, :, :, :].copy()
        frame_net = torch.from_numpy(frame_net).to(device)

        with torch.no_grad():
            features = model(frame_net).cpu().numpy()
            frames_features.append(features)
            frame_times.append(t)
            cv2.imwrite(f'{args.output_dir}/{frame_id:04d}.jpg', frame_image)
            frame_id += 1

    with open(f'{args.output_dir}/frame_times.json', 'w') as f:
        json.dump(frame_times, f)

    with open(f'{args.output_dir}/frames_features.npy', 'wb') as f:
        np.save(f, np.array(frames_features))

    print(f'Extracted {frame_id} frames.')


if __name__ == '__main__':
    main()
