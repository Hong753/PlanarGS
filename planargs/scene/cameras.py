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

import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from planargs.common_utils.general_utils import PILtoTorch
from planargs.common_utils.graphics_utils import getWorld2View2, getProjectionMatrix, get_k, ThickenLines

#----------------------------------------------------------------------------

class Camera(nn.Module):
    def __init__(
            self, colmap_id, R, T, FoVx, FoVy, resolution, path, 
            params, image_name, uid, 
            trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
            data_device = "cuda",
        ):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = resolution[0]
        self.image_height = resolution[1]
        
        image_path = os.path.join(path, "images", image_name[0] + image_name[1])
        depth_path = os.path.join(path, "depth", image_name[0] + ".png")
        depthconf_path = os.path.join(path, "mask", image_name[0] + ".png")
        priornormal_path = os.path.join(path, "planargs_priors", "normal", image_name[0] + ".npy")
        planarmask_path = os.path.join(path, "planargs_priors", "planar_mask", image_name[0] + ".npy")
        
        # image_path = os.path.join(path, "images", image_name[0] + image_name[1])
        # geomprior_folder = os.path.join(path, "planargs_priors", )
        # depthconf_path = os.path.join(geomprior_folder, "resized_confs", image_name[0] + ".npy")
        # planarmask_path = os.path.join(path, "planarprior/mask", image_name[0] + ".npy")
        # weights_path = os.path.join(geomprior_folder, "depth_weights.json")

        original_image = Image.open(image_path)
        resized_image = original_image.resize(resolution)
        self.gt_image = PILtoTorch(resized_image, resolution)[:3, ...].clamp(0.0, 1.0).to(self.data_device) # (3, H, W)
        
        canny_mask = cv2.Canny(np.array(resized_image), params.canny_thresh[0], params.canny_thresh[1])/255.  
        canny_masker = torch.from_numpy(canny_mask).clamp(0.0, 1.0).to(self.data_device)
        self.canny_mask = 1 - ThickenLines(canny_masker, kernel_size=5)
        
        # Load depth
        if os.path.exists(depth_path):
            depth_mm_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = depth_mm_u16.astype(np.float32) / 1000.0
            depth = cv2.resize(depth, resolution, interpolation=cv2.INTER_NEAREST)
            self.priordepth = torch.from_numpy(depth).to(self.data_device)
        else:
            self.priordepth = None
        
        # Load depth confidence mask
        if os.path.exists(depthconf_path):
            depthconf_u16 = cv2.imread(depthconf_path, cv2.IMREAD_UNCHANGED)
            depthconf = (depthconf_u16.astype(np.float32) / 10_000) > params.conf_thresh
            depthconf = cv2.resize(depthconf.astype(np.float32), resolution, interpolation=cv2.INTER_NEAREST)
            self.depth_conf = torch.from_numpy(depthconf).clamp(0.0, 1.0).to(self.data_device)
            self.depth_weight = self.depth_conf.mean()
        else:
            self.depth_conf = None
            self.depth_weight = None
        
        # Load normal prior
        if os.path.exists(priornormal_path):
            priornormal = np.load(priornormal_path)
            priornormal = cv2.resize(priornormal, resolution, interpolation=cv2.INTER_NEAREST)
            self.priornormal = torch.from_numpy(priornormal).to(self.data_device).permute(2, 0, 1)
        else:
            self.priornormal = None
                
        # Load LP3 prior
        if os.path.exists(planarmask_path):
            planar_mask = np.load(planarmask_path)
            planar_mask = cv2.resize(planar_mask, resolution, interpolation=cv2.INTER_NEAREST)
            self.planarmask = torch.from_numpy(planar_mask).to(self.data_device).to(torch.int64)   
        else:
            self.planarmask = None

        self.K, self.inv_K = get_k(FoVx, FoVy, self.image_height, self.image_width, scale)
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()  #4x4
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()  #4x4
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)  #4x4
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1).float().cuda()
        return rays_d

#----------------------------------------------------------------------------

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

#----------------------------------------------------------------------------