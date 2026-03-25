#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
import torch.nn.functional as F


def Pointscam2Depth(K, points, size, depth=False, errors=None):  
    """
    Args:
        K: (3, 3) torch.Tensor
        points: (3, N) torch.Tensor
    Return: 
        depthmap: (H, W) torch.Tensor
        weights:  (H, W) torch.Tensor
    """
    height, width = size
    cam_coord = K @ points
    z = cam_coord[2]
    x = cam_coord[0] / z
    y = cam_coord[1] / z

    valid_mask = (
        (z > 0) &
        (x >= 0) & (x <= width - 1) &
        (y >= 0) & (y <= height - 1)
    )
    valid_idx = torch.where(valid_mask)[0]
    pts_depths = z[valid_idx]

    x_valid = x[valid_idx]
    y_valid = y[valid_idx]
    h = torch.round(y_valid).long().clamp(0, height - 1)
    w = torch.round(x_valid).long().clamp(0, width - 1)
    depthmap = torch.zeros((height, width), device=points.device)

    # if depth=False, return mask
    if depth:
        depthmap[h, w] = pts_depths
        if errors is not None:
            weight = torch.zeros((height, width), device=points.device)
            weight[h, w] = 1.0 / errors[valid_idx].float()
            weights = weight / weight.max()
            return depthmap, weights
        else:
            return depthmap, torch.ones((height, width), device=points.device)
    else:
        depthmap[h, w] = 1
        return depthmap, None


def Depth2Pointscam(depth, inv_K, mask=None):
    """
    Convert depth map to 3D points in camera coordinates (torch version).

    Args:
        depth: (H, W) torch.Tensor
        inv_K: (3, 3) torch.Tensor
        mask: (H, W) torch.Tensor or None, >0 indicates valid pixels

    Returns:
        points_3d: (N, 3) torch.Tensor
    """
    device = depth.device
    dtype = depth.dtype
    H, W = depth.shape

    if mask is None:
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        x = x.reshape(-1)
        y = y.reshape(-1)
    else:
        y, x = torch.where(mask > 0)

    ones = torch.ones_like(x, dtype=dtype)
    coords_h = torch.stack(
        [x.to(dtype), y.to(dtype), ones],
        dim=0
    )  # (3, N)
    norm_coords = inv_K @ coords_h  # (3, N)
    points_3d = (norm_coords * depth[y, x]).T  # (N, 3)

    return points_3d, coords_h


def pcd2normal(xyz):
    """
    xyz: (H, W, 3) torch.Tensor
    return: (H, W, 3) normal
    """
    left  = xyz[1:-1, :-2]
    right = xyz[1:-1,  2:]
    top   = xyz[:-2,  1:-1]
    bottom= xyz[2:,   1:-1]

    v1 = right - left
    v2 = top - bottom

    normal = torch.cross(v1, v2, dim=-1)
    normal = torch.nn.functional.normalize(normal, dim=-1)
    normal = torch.nn.functional.pad(
        normal.permute(2,0,1),
        (1,1,1,1),
        mode='replicate'
    ).permute(1,2,0)

    return normal


def NormalFromDepth(depth, inv_K):
    # depth: (H, W), intrinsic_matrix: (3, 3)
    # xyz_normal: (H, W, 3)
    xyz_cam, _ = Depth2Pointscam(depth, inv_K)
    xyz_cam = xyz_cam.view(depth.shape[0], depth.shape[1], 3) # (H, W, 3)        
    xyz_normal = pcd2normal(xyz_cam)

    return xyz_normal


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    errors : np.array = None


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def get_k(Fovx, Fovy, H, W, scale=1.0, device="cuda"):
    Fx = fov2focal(Fovx, W)
    Fy = fov2focal(Fovy, H)
    Cx = 0.5 * W
    Cy = 0.5 * H
    K = [[Fx / scale, 0, Cx / scale],
        [0, Fy / scale, Cy / scale],
        [0, 0, 1]]
    
    inv_K = [[scale/Fx, 0, -Cx/Fx],
            [0, scale/Fy, -Cy/Fy],
            [0, 0, 1]]
    return torch.tensor(K).to(device), torch.tensor(inv_K).to(device)


def RenderDistance(points3D, normals):
    distance = (normals * points3D).sum(-1).abs()
    return distance


def ThickenLines(image, kernel_size=3):
    image = image.float()
    # (H, W) -> (1, 1, H, W)
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(1)

    padding = kernel_size // 2
    # dilation = max pooling
    thickened = F.max_pool2d(
        image,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )

    return thickened.squeeze(0).squeeze(0)