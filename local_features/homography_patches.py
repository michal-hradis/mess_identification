import cv2
import numpy as np
import logging
import os
import time
#from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights, raft_large, Raft_Large_Weights
import torchvision.transforms as T
import torch
import lmdb

def extract_patch(img: np.ndarray, kp: cv2.KeyPoint, patch_size: int) -> np.ndarray:
    """
    Extracts a patch from the image using the keypoint's scale and orientation.
    The patch is a rotated and scaled crop such that the keypoint's neighborhood
    is normalized to a canonical coordinate system.

    Parameters:
        img (np.ndarray): The input image.
        kp (cv2.KeyPoint): The SIFT keypoint.
        patch_size (int): The desired output patch size (in pixels).

    Returns:
        patch (np.ndarray): The extracted patch of shape (patch_size, patch_size, channels),
                            or None if the patch goes out of image boundaries.
    """
    # Keypoint parameters: (x, y), size, and angle.
    x, y = kp.pt
    # The size attribute is the diameter of the meaningful region.
    # We use it to compute a scaling factor so that the region maps to patch_size.
    scale = kp.size / patch_size  # pixels in image per patch pixel

    # The keypoint angle is in degrees. We convert it to radians.
    theta = np.deg2rad(kp.angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    # Build the affine transformation matrix M that maps coordinates (u,v)
    # from the canonical patch coordinate system (with center at (patch_size/2, patch_size/2))
    # to the image coordinates.
    # The transform is: [u,v] -> [x, y] + scale * R * ([u - patch_size/2, v - patch_size/2])
    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = scale * np.array([[cos_theta, -sin_theta],
                                 [sin_theta, cos_theta]])
    center_patch = np.array([patch_size / 2, patch_size / 2], dtype=np.float32)
    M[:, 2] = np.array([x, y], dtype=np.float32) - M[:, :2].dot(center_patch)

    # Compute the transformed coordinates of the patch corners
    corners = np.array([[0, 0], [patch_size, 0], [patch_size, patch_size], [0, patch_size]], dtype=np.float32)
    transformed_corners = cv2.transform(np.array([corners]), M)[0]

    h, w = img.shape[:2]
    # Check if any corner lies outside the image boundaries.
    if (transformed_corners[:, 0].min() < 0 or transformed_corners[:, 0].max() >= w or
            transformed_corners[:, 1].min() < 0 or transformed_corners[:, 1].max() >= h):
        return None

    M_inverse = cv2.invertAffineTransform(M)
    # Warp the image to extract the patch using the computed affine transform.
    patch = cv2.warpAffine(img, M_inverse, (patch_size, patch_size), flags=cv2.INTER_LINEAR)
    return patch

def compute_homography(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compute the homography matrix that maps points from img1 to img2.

    Parameters:
        img1 (np.ndarray): First RGB image.
        img2 (np.ndarray): Second RGB image.

    Returns:
        np.ndarray: The 3x3 homography matrix.
    """
    # Convert images to grayscale.
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    for sigma in [1.6, 2.5, 3.5, 1.4, 1.2]:
        # Initialize SIFT detector.
        sift = cv2.SIFT_create(20000, 3, sigma=sigma)

        t1 = time.time()
        # Find keypoints and descriptors.
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        t2 = time.time()

        # Use pytorch to compute the matches
        desc1_t = torch.from_numpy(des1).to('cuda').float()
        desc2_t = torch.from_numpy(des2).to('cuda').float()
        desc1_t = desc1_t / desc1_t.norm(dim=1)[:, None]
        desc2_t = desc2_t / desc2_t.norm(dim=1)[:, None]
        sim = torch.mm(desc1_t, desc2_t.t())
        matches = torch.topk(sim, k=2, dim=1)

        # Apply Lowe's ratio test to filter good matches.
        good_matches = []
        sim_threshold = 0.9
        match_indices = matches.indices.cpu().numpy()
        match_similarities = matches.values.cpu().numpy()
        for i in range(match_indices.shape[0]):
            if match_similarities[i, 0] * sim_threshold > match_similarities[i, 1]:
                good_matches.append(cv2.DMatch(i, match_indices[i, 0], 1 - match_similarities[i, 0]))
        t3 = time.time()

        # Prepare source and destination points for homography estimation.
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate homography using RANSAC.
        if len(src_pts) > 30:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 15.0)
        else:
            H = None
            mask = np.zeros(1, dtype=np.uint8)
        t4 = time.time()
        logging.info(f"MATCH ransac_matches: {mask.sum()}, good_matches: {len(good_matches)}, kp1: {len(kp1)}, kp2: {len(kp2)} in {t2 - t1:.2f}, {t3 - t1:.2f}, {t4 - t1:.2f} s")
        if mask.sum() > 30:
            break
        else:
            H = None

    return H


def extract_homography_patches(img1: np.ndarray, img2: np.ndarray, patch_size: int = 32) -> list:
    """
    Given two RGB images, compute SIFT features, find matching features,
    estimate a homography with RANSAC, and extract patch pairs using each keypoint's
    estimated scale and orientation for normalization.

    Parameters:
        img1 (np.ndarray): First RGB image.
        img2 (np.ndarray): Second RGB image.
        patch_size (int): Size of the canonical square patch to extract (default 32).

    Returns:
        List of tuples: Each tuple contains (patch_from_img1, patch_from_img2)
                        for each valid matching feature pair.
    """

    target_size = 1900
    img1 = cv2.resize(img1, (target_size, target_size))
    img2 = cv2.resize(img2, (target_size, target_size))

    #img1 = cv2.resize(img1, (640, 480))
    #img2 = cv2.resize(img2, (640, 480))

    # Convert images to grayscale for SIFT detection.
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Initialize SIFT detector.
    sift = cv2.SIFT_create(20000, 4, sigma=5.5)

    # Detect keypoints and compute descriptors.
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    for kp in kp1:
        kp.size *= 3
    for kp in kp2:
        kp.size *= 3

    logging.info(f"Found keypoints: {len(kp1)}, {len(kp2)}")

    # show the keypoints
    #img1 = cv2.drawKeypoints(img1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #img2 = cv2.drawKeypoints(img2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow('Keypoints 1', img1)
    #cv2.imshow('Keypoints 2', img2)
    #cv2.waitKey(0)

    # Use BFMatcher with L2 norm (appropriate for SIFT).
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter matches.
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.90 * n.distance:
            good_matches.append(m)

    logging.info(f"Initial matches: {len(knn_matches)}, good matches: {len(good_matches)}")

    if len(good_matches) < 4:
        print("Not enough good matches were found.")
        return []

    # Prepare points for homography estimation.
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate homography using RANSAC.
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
    if H is None:
        print("Homography could not be computed.")
        return []

    logging.info(f"Homography consistent matches: {mask.sum()}")

    # The mask indicates which matches are inliers.
    mask = mask.ravel().tolist()
    patch_pairs = []

    # Loop over the good matches that are inliers.
    for i, m in enumerate(good_matches):
        if mask[i]:
            patch1 = extract_patch(img1, kp1[m.queryIdx], patch_size)
            patch2 = extract_patch(img2, kp2[m.trainIdx], patch_size)
            # Only add the pair if both patches were successfully extracted.
            if patch1 is not None and patch2 is not None:
                patch_pairs.append((patch1, patch2))

    return patch_pairs


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
        ]
    )
    batch = transforms(batch)
    return batch


def main():
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser(description='Extract homography patches from two images.')
    parser.add_argument('--images', nargs="+", type=str, required=True, help='Paths to the images to process.')
    parser.add_argument('--patch-size', type=int, default=80, help='Size of the extracted patches.')
    parser.add_argument('--output-path', type=str, default='./output/', help='Path to save the extracted patches.')
    parser.add_argument('--output-prefix', type=str, default='patch', help='Prefix for the output patch filenames.')
    parser.add_argument('--output-lmdb', type=str, default='./output.lmdb', help='Path to save the extracted patches.')
    parser.add_argument("--grid-step", type=int, default=32, help="Step size for the grid points.")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    reference_image = cv2.imread(args.images[0])
    reference_image = cv2.resize(reference_image, (reference_image.shape[1] // 2, reference_image.shape[0] // 2),
                                 interpolation=cv2.INTER_AREA)

    DEVICE = 'cuda'
    #of_model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(DEVICE)
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(DEVICE)
    of_model = of_model.eval()


    grid_points = np.array([[x, y] for x in range(0, reference_image.shape[1], args.grid_step)
                            for y in range(0, reference_image.shape[0], args.grid_step)], dtype=np.float32)

    # Create a LMDB database to store the patches
    if args.output_lmdb:
        env = lmdb.open(args.output_lmdb, map_size=1024 ** 4)



    for image_id, img_path in enumerate(args.images[1:]):
        img2 = cv2.imread(img_path)
        img2 = cv2.resize(img2, (img2.shape[1] // 2, img2.shape[0] // 2), interpolation=cv2.INTER_AREA)

        H = compute_homography(img2, reference_image)

        if H is None:
            logging.info(f"Failed to compute homography for {img_path}")
            continue

        # Map image 2 to image 1 using the computed homography.
        img2_warped = cv2.warpPerspective(img2, H, reference_image.shape[:2][::-1])
        of_images = [reference_image, img2_warped]

        of_scale = 0.5
        of_images = [cv2.resize(img, (int(img.shape[1] * of_scale), int(img.shape[0] * of_scale))) for img in of_images]

        img = np.stack(of_images, 0)
        img_torch = preprocess(torch.from_numpy(np.transpose(img, (0, 3, 1, 2)) / 255.0)).to(DEVICE)
        img0_torch = img_torch[0:1]
        img1_torch = img_torch[1:2]
        with torch.no_grad():
            # pad the input images to the next multiple of 8
            h, w = img0_torch.shape[-2:]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            img0_torch = torch.nn.functional.pad(img0_torch, (0, pad_w, 0, pad_h))
            img1_torch = torch.nn.functional.pad(img1_torch, (0, pad_w, 0, pad_h))
            predicted_flows = of_model(img0_torch, img1_torch, num_flow_updates=8)
            predicted_flows = predicted_flows[-1][0]
            predicted_flows = predicted_flows[:, :h, :w]
            predicted_flows = predicted_flows.cpu().numpy()

        flow_x = predicted_flows[0, :, :] / of_scale
        flow_y = predicted_flows[1, :, :] / of_scale
        flow_x = cv2.resize(flow_x, (reference_image.shape[1], reference_image.shape[0]))
        flow_y = cv2.resize(flow_y, (reference_image.shape[1], reference_image.shape[0]))
        # warp the image using the predicted flow
        map_x = (np.arange(0, reference_image.shape[1], 1)[np.newaxis, :] + flow_x).astype(np.float32)
        map_y = (np.arange(0, reference_image.shape[0], 1)[:, np.newaxis] + flow_y).astype(np.float32)
        img2_warped_of = cv2.remap(img2_warped, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        # transform the grid points using the predicted flow and homography into the second image
        grid_points_2 = np.stack(
            [map_x[grid_points[:, 1].astype(int), grid_points[:, 0].astype(int)],
             map_y[grid_points[:, 1].astype(int), grid_points[:, 0].astype(int)]], axis=1)
        H_inv = np.linalg.inv(H)
        grid_points_2 = cv2.perspectiveTransform(grid_points_2[np.newaxis, :, :], H_inv)[0]
        # render the grid points on the second image

        # crop patches from img2
        if args.output_lmdb:
            txn = env.begin(write=True)
            patch_size = args.patch_size
            for i, (x, y) in enumerate(grid_points_2):
                x1 = int(x + 0.5) - patch_size // 2
                y1 = int(y + 0.5) - patch_size // 2
                x2 = x1 + patch_size
                y2 = y1 + patch_size
                if x1 < 0 or y1 < 0 or x2 >= img2.shape[1] or y2 >= img2.shape[0]:
                    continue
                patch = img2[y1:y2, x1:x2]
                output_file = f'{args.output_prefix}-{i:04d}_{image_id:02d}.jpg'
                patch_img = cv2.imencode('.jpg', patch, [int(cv2.IMWRITE_JPEG_QUALITY), 98])[1].tobytes()
                txn.put(output_file.encode(), patch_img)
            txn.commit()
            logging.info(f"Saved patches for {img_path}, {len(grid_points_2)} patches")

        for x, y in grid_points_2:
            cv2.circle(img2, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.imshow('Image 2', img2)
        cv2.waitKey(100)

        composed_image = cv2.addWeighted(reference_image, 0.35, img2_warped_of, 0.65, 0)
        cv2.imwrite(f'composed_{os.path.basename(img_path)}', composed_image)

        #images = [reference_image, img2_warped_of, composed_image]
        #for i in range(40000):
        #    cv2.imshow(f'Image', images[i % len(images)])
        #    key = cv2.waitKey(0)
        #    if key == 27:
        #        exit()
        #    if key == ord('n'):
        #        break


    """patch_pairs = extract_homography_patches(reference_image, img_path, args.patch_size)
    patches1 = [pair[0] for pair in patch_pairs[:32]]
    patches2 = [pair[1] for pair in patch_pairs[:32]]
    row1 = np.concatenate(patches1, axis=1)
    row2 = np.concatenate(patches2, axis=1)
    collage = np.concatenate([row1, row2], axis=0)
    cv2.imshow('Homography Patches', collage)
    cv2.waitKey(0)"""


if __name__ == '__main__':
    main()

