import argparse
import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize image features using T-SNE')
    parser.add_argument('--input-path', type=str, required=True,
                        help='Path to a directory containing frames_features.npy, images with filenames '
                             '00001.jpg, 00002.jpg, etc. and json files containing list of frame timestamps.')
    return parser.parse_args()


def load_features(input_path):
    # Load features
    features = np.load(f'{input_path}/frames_features.npy')[:, 0, :]
    # Load image paths
    #image_paths = [f'{input_path}/{i:05d}.jpg' for i in range(1, features.shape[0] + 1)]
    image_paths = [os.path.join(input_path, f'{i:04d}.jpg') for i in range(1, features.shape[0] + 1)]
    return features, image_paths


def plot_feature_similarity_matrix(features: np.ndarray):
    # Compute cosine similarity between features
    similarity_matrix = np.dot(features, features.T)

    # Plot similarity matrix
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Feature Similarity Matrix')
    plt.show()


def plot_tsne_images(X_tsne, image_paths):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)

    for (x, y), path in zip(X_tsne, image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        imagebox = OffsetImage(img, zoom=0.4)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)

    ax.set_title('T-SNE Visualization of Image Features')
    plt.show()


def plot_tsne_frame_timestamps(X_tsne, frame_timestamps):
    """ Render points color-coded with frame timestamps """

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=frame_timestamps, cmap='plasma', alpha=0.8)

    #for (x, y), timestamp in zip(X_tsne, frame_timestamps):
    #    ax.text(x, y, timestamp, fontsize=12)

    ax.set_title('T-SNE Visualization of Image Features')
    plt.show()


def play_video_fast_adaptive(features: np.ndarray, video_frames, show_frame_count=100, temperature=0.5):
    """ Play video at a faster rate based on feature similarity. Plays faster when frames are similar. """

    # Compute cosine similarity between features
    similarity_matrix = np.dot(features, features.T) - 1
    similarity_matrix = np.exp(similarity_matrix / temperature)
    distance = 1 - similarity_matrix

    consecutive_frame_distance = np.diag(distance, k=1)
    distance_sum = np.cumsum(consecutive_frame_distance)

    show_frame_step = distance_sum[-1] / show_frame_count
    shown_frame_count = 0
    current_frame = 0
    while current_frame < len(video_frames):
        cv2.imshow('Video', video_frames[current_frame])
        cv2.waitKey(100)
        shown_frame_count += 1
        while distance_sum[current_frame] < shown_frame_count * show_frame_step:
            current_frame += 1

def play_video(video_frames):
    for frame in video_frames:
        cv2.imshow('Video', frame)
        cv2.waitKey(100)


def main():
    args = parse_args()
    features, image_paths = load_features(args.input_path)

    plot_feature_similarity_matrix(features)

    # Apply T-SNE to reduce dimensions to 2
    tsne = TSNE(n_components=2, random_state=41)
    X_tsne = tsne.fit_transform(features)

    plot_tsne_frame_timestamps(X_tsne, np.arange(1, features.shape[0] + 1))

    plot_tsne_images(X_tsne, image_paths)


    video_frames = [cv2.imread(image_path) for image_path in image_paths][:-1]
    video_frames = [f[..., ::-1] for f in video_frames]

    # Play video
    play_video(video_frames)

    for temperature in [10, 5, 2, 1, 0.5, 0.3, 0.1]:
        print(temperature)
        play_video_fast_adaptive(features, video_frames, temperature=temperature)




    # Apply T-SNE to reduce dimensions to 2
    tsne = TSNE(n_components=2, random_state=41)
    X_tsne = tsne.fit_transform(features)

    plot_tsne_frame_timestamps(X_tsne, np.arange(1, features.shape[0] + 1))

    plot_tsne_images(X_tsne, image_paths)





if __name__ == '__main__':
    main()
