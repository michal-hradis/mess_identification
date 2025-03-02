import decord
import cv2
import numpy as np
import argparse
import onnxruntime as ort
from tqdm import tqdm
from sklearn.cluster import KMeans

def parse_arguments():
    parser = argparse.ArgumentParser(description="Reads frames from video and clusters them based on frame descriptors computed by a neural network.")
    parser.add_argument('--video', required=True, help='Video file file to process.', )
    parser.add_argument('--model', required=True, help='ONNX model file.' )
    parser.add_argument('--resolution', default=128, type=int, help='Resolution of frames.')
    parser.add_argument('--frame-count', default=2000, type=int, help='Number of frames to process.')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    video = decord.VideoReader(args.video)
    print(f'Video frames: {len(video)}')

    ort_session = ort.InferenceSession(args.model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    test_data = np.zeros((1, 3, args.resolution, args.resolution), dtype=np.uint8)
    outputs = ort_session.run(None, {'input_image': test_data})[0]
    emb_dim = outputs.shape[1]
    print(f'Embedding dimension: {emb_dim}')

    frame_count = len(video)
    frames_to_process = int(min(frame_count / 4, args.frame_count))
    print(f'Video frames: {frame_count}')
    print(f'Frames to process: {frames_to_process}')
    frame_to_process = np.linspace(0, frame_count-1, frames_to_process, dtype=int)

    print('Reading video frames...')
    video_reader = decord.VideoReader(args.video, width=args.resolution, height=args.resolution)
    frames = video_reader.get_batch(frame_to_process).asnumpy()

    print('Computing frame descriptors...')
    descriptors = []

    for frame in tqdm(frames):
        frame = np.transpose(frame, (2, 0, 1))[np.newaxis, ::-1]
        embedding = ort_session.run(None, {'input_image': frame})
        descriptors.append(embedding[0])

    descriptors = np.concatenate(descriptors, axis=0)
    print(f'Descriptors shape: {descriptors.shape}')

    print('Clustering frames...')
    while True:
        cluster_count = input('Enter number of clusters: ')
        if not cluster_count.isdigit():
            print('Invalid input. Please enter a number.')
            continue
        cluster_count = int(cluster_count)

        kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(descriptors)
        labels = kmeans.labels_

        collage_width = 8
        for i in range(cluster_count):
            print(f'Cluster {i}: {np.sum(labels == i)} frames')
            cluster_images = frames[labels == i]
            lines = [np.concatenate(cluster_images[i:i+collage_width], axis=1) for i in range(0, len(cluster_images), collage_width)]
            if lines[-1].shape[1] < collage_width * args.resolution:
                lines[-1] = np.pad(lines[-1], ((0, 0), (0, collage_width * args.resolution - lines[-1].shape[1]), (0, 0)))
            collage = np.concatenate(lines, axis=0)
            cv2.imshow(f'Cluster', collage[:, :, ::-1])
            key = cv2.waitKey()
            if key == 27:
                break


if __name__ == '__main__':
    main()

