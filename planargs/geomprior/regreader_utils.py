import os
from PIL import Image
import numpy as np
from .align_opt import OptimizeGroupDepth
import torch
import cv2
from planargs.scene.ply_loader import CameraInfo
from dataclasses import dataclass
from planargs.common_utils.graphics_utils import get_k, NormalFromDepth, Pointscam2Depth
from planargs.planar.visualize import visualDepth, visualNorm


@dataclass
class DepthInfo:
    cam_info: CameraInfo
    depth_aligned: np.array
    prior_normal: np.array
    depth_conf: np.array
    depth_weight: float


def AlignGroupDepth(group_cam_infos, depth_list, pcd, conf_list, vis_path, prep, device="cuda"):
    depthmap_list = []
    depths_list = []
    colmapweight_list = []
    resized_conf_list = []
    for idx in range(len(group_cam_infos)):
        cam_info = group_cam_infos[idx]
        height = cam_info.size[1]
        width = cam_info.size[0]

        # Select only the points visible in the current view.
        valid_point3D_ids = cam_info.points3d_ids[cam_info.points3d_ids != -1]
        count = 0
        xyz = []
        errors = []
        for pid_3d in pcd[:,0]:
            for pid in valid_point3D_ids:
                if pid_3d == pid:   
                    errors.append(pcd[count, 1])
                    xyz.append(pcd[count, 2:5])
                    continue
            count += 1
        errors = torch.from_numpy(np.array(errors)).to(device)
        xyz = torch.from_numpy(np.array(xyz)).float().to(device)

        colmap_weight = torch.zeros((height, width), device=device)
        K, inv_K = get_k(cam_info.FovX, cam_info.FovY, height, width)
        R = torch.from_numpy(cam_info.R).float().to(device)
        T = torch.from_numpy(cam_info.T).float().to(device)
        points3d = R.t() @ xyz.t() + T.view(3, 1)  # (3, N)

        # Assign weights to the point cloud based on COLMAP quality.
        depthmap, colmap_weight = Pointscam2Depth(K, points3d, size=(height, width), depth=True, errors=errors)
        depthmap_list.append(depthmap)  # d_sparse
        colmapweight_list.append(colmap_weight) 
       
        depth = depth_list[idx] 
        # Scale adjustment for depth map generateds by the feedforward model.
        if depth.shape != (height, width):
            depth_list[idx] = cv2.resize(depth, cam_info.size, interpolation=cv2.INTER_LANCZOS4)
            if conf_list != []:
                conf = conf_list[idx]
                conf = cv2.resize(conf, cam_info.size, interpolation=cv2.INTER_LANCZOS4)
                resized_conf_list.append(conf)
            else:
                resized_conf_list.append(np.ones((height, width)))

        depths_list.append(torch.from_numpy(depth_list[idx]).to(device))  # d_dense
                 
    depth_aligned_list, depth_param, alignoff_loss = OptimizeGroupDepth(source=depths_list, target=depthmap_list, weight=colmapweight_list, prep=prep) 

    depthinfo_list = []
    for idx in range(len(group_cam_infos)):
        cam_info = group_cam_infos[idx]
        K, inv_K = get_k(cam_info.FovX, cam_info.FovY, cam_info.size[1], cam_info.size[0])
        prior_depth = depth_aligned_list[idx].clone().detach()
        prior_normal = NormalFromDepth(prior_depth, inv_K)
        if vis_path is not None:
            vis_depth_path = os.path.join(vis_path, "depth_vis")
            vis_normal_path = os.path.join(vis_path, "normal_vis")
            os.makedirs(vis_depth_path, exist_ok=True)
            os.makedirs(vis_normal_path, exist_ok=True)
            visualDepth(prior_depth, vis_depth_path, cam_info.image_name[0])
            visualNorm(prior_normal.permute(2, 0, 1), vis_normal_path, cam_info.image_name[0])

        depth = prior_depth.cpu().numpy()
        normal = prior_normal.cpu().numpy()
        depthinfo_list.append(DepthInfo(cam_info=group_cam_infos[idx], depth_aligned=depth, prior_normal=normal, 
                                        depth_conf=resized_conf_list[idx], depth_weight=alignoff_loss))

    return depthinfo_list, depth_param


def LoadGroupDepth(group_cam_infos, depth_folder, conf_folder, pcd, vis_path, prep):
    depth_list = []
    conf_list = []

    for idx in range(len(group_cam_infos)):
        cam_info = group_cam_infos[idx]
        image_name = cam_info.image_name[0]
        depth_path = os.path.join(depth_folder, image_name)

        if os.path.exists(conf_folder):
            conf_path = os.path.join(conf_folder, image_name)
            depth_conf = np.load(conf_path + ".npy")
            conf_list.append(depth_conf)

        if os.path.exists(depth_path + ".npy"):
            depth = np.load(depth_path + ".npy")
        elif os.path.exists(depth_path + ".png"):
            depth = Image.open(depth_path + ".png")
            depth = np.array(depth).astype(np.float32)
        elif os.path.exists(depth_path + ".jpg"):
            depth = Image.open(depth_path + ".jpg")
            depth = np.array(depth).astype(np.float32)
        else:
            assert False, "Could not recognize depth type!"

        depth_list.append(depth)

    return AlignGroupDepth(group_cam_infos, depth_list, pcd, conf_list, vis_path, prep)





