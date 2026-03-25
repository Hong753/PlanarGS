import numpy as np
import cv2
import torch
from planargs.common_utils.graphics_utils import RenderDistance, ThickenLines, Depth2Pointscam              


def merge_similar(masks, image_rgb, color_thresh=50):
    avg_colors = []
    for mask in masks:
        if mask.sum() > 0:
            # image_rgb[mask] -> (N, 3)
            avg_color = image_rgb[mask.bool()].mean(dim=0)
        else:
            avg_color = torch.zeros(3, device=image_rgb.device)
        avg_colors.append(avg_color)

    merged_masks = []
    used = torch.zeros(len(masks), dtype=torch.bool)

    for i in range(len(masks)):
        if used[i]:
            continue
        current_mask = masks[i].clone().bool()

        for j in range(i + 1, len(masks)):
            if used[j]:
                continue
            color_diff = torch.norm(avg_colors[i] - avg_colors[j])
            if color_diff < color_thresh:
                current_mask |= masks[j].bool()
                used[j] = True

        merged_masks.append(current_mask.to(torch.uint8))

    return merged_masks


def kmeans_torch(x, num_clusters, num_iters=10, device='cuda'):
    """
    x: (N, D) float tensor on CUDA
    Returns: cluster labels (N,), cluster centers (K, D)
    """
    N, D = x.shape
    # Randomly initialize K cluster centers
    indices = torch.randperm(N, device=device)[:num_clusters]
    centers = x[indices]

    for _ in range(num_iters):
        # Compute distances (N, K)
        dist = torch.cdist(x, centers, p=2) 
        labels = dist.argmin(dim=1)  

        # Update centers
        for k in range(num_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centers[k] = x[mask].mean(dim=0)

    return labels, centers


def SplitPic(image_rgb, num_clusters=5):
    """
    Returns:
        list of torch.tensor: A list of masks corresponding to each cluster.
    """
    pixels = image_rgb.reshape((-1, 3)).float()
    labels, _ = kmeans_torch(pixels, num_clusters)
    cluster_map = labels.reshape(image_rgb.shape[:2]) + 1  # +1 to distinguish from background
    
    masks = []
    for cluster_idx in range(1, num_clusters + 1):
        mask = (cluster_map == cluster_idx).to(torch.uint8)  
        if mask.sum() > 2000:
            masks.append(mask)
    masks = merge_similar(masks, image_rgb)

    return masks


def MaxConnect(masker, thresh_1=2000, thresh_2=700):
    """
    Extract the largest connected region from the mask.
    """
    mask = masker.cpu().numpy().astype(np.uint8)
    device = masker.device
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 0: 
            components.append((area, i))
    
    components.sort(key=lambda x: x[0], reverse=True)
    largest_area, largest_label = components[0]
    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 1

    if len(components) < 2:
        if largest_area < thresh_1:
            return None, None
        else:
            return torch.tensor(largest_mask).int().to(device), None

    second_largest_area, second_largest_label = components[1] 
    second_mask = np.zeros_like(mask)
    second_mask[labels == second_largest_label] = 1

    if second_largest_area < thresh_2:
        return torch.tensor(largest_mask).int().to(device), None
    else:
        return torch.tensor(largest_mask).int().to(device), torch.tensor(second_mask).int().to(device)


def MaskDistance(depth, normal, inv_K, thresh=0.05):
    points3D, _ = Depth2Pointscam(depth, inv_K)
    points3D = points3D.view(normal.shape[0], normal.shape[1], 3)
    distance = RenderDistance(points3D, normal)

    prior_distance = (distance - distance.min()) / (distance.max() - distance.min() + 1e-8)
    distance_mask = (prior_distance > thresh).float()
    distance_mask = 1 - ThickenLines(1 - distance_mask)  
    
    return distance_mask   
