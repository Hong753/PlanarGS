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
import sys
from planargs.scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary
from planargs.common_utils.graphics_utils import focal2fov
from planargs.scene.ply_loader import SceneInfo, CameraInfo, getNerfppNorm, storePly, fetchPly
import numpy as np
import json
import torch


def load_poses(pose_path, num):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for i in range(num):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3,3] = c2w[:3,3] * 10.0
        w2c = np.linalg.inv(c2w)
        w2c = w2c
        poses.append(w2c)
    poses = np.stack(poses, axis=0)
    return poses


def SingleReadColmap(idx, key, cam_extrinsics, cam_intrinsics, path):
    sys.stdout.write('\r')
     # the exact output you're looking for:
    sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
    sys.stdout.flush()

    extr = cam_extrinsics[key]
    intr = cam_intrinsics[extr.camera_id]
    height = intr.height 
    width = intr.width 

    uid = intr.id  
    R = np.transpose(qvec2rotmat(extr.qvec))
    T = np.array(extr.tvec)

    if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[0]
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)
    elif intr.model=="PINHOLE":
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
    else:
        assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

    image_name = os.path.splitext(extr.name)
    points3d_ids = extr.point3D_ids
        
    cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
                        path=path, image_name=image_name,
                        size=(width, height), points3d_ids=points3d_ids)
        
    return cam_info


def readColmapCameras(cam_extrinsics, cam_intrinsics, path):
    cam_infos = []

    for idx, key in enumerate(cam_extrinsics): 
        cam_info = SingleReadColmap(idx, key, cam_extrinsics, cam_intrinsics, path)
        cam_infos.append(cam_info)
        torch.cuda.empty_cache()
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfo(path, eval, llffhold=8):  
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
            
    N = len([f for f in os.listdir(os.path.join(path, "images"))])
    
    js_file = f"{path}/train_test_lists.json"

    if eval:
        if os.path.exists(js_file):
            train_list = []
            test_list = []
            with open(js_file) as file:
                meta = json.load(file)
                train_list = [os.path.splitext(file)[0] for file in meta["train"]]
                test_list = [os.path.splitext(file)[0] for file in meta["test"]]
                print(f"train_list {len(train_list)}, test_list {len(test_list)}")

        else:
            train_list = [idx for idx in range(N) if idx % llffhold != 0]
            test_list = [idx for idx in range(N) if idx % llffhold == 0]
    else:
        train_list = [idx for idx in range(N)]


    ply_path = os.path.join(path, "sparse/points3D.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    if os.path.exists(bin_path):
        xyz, rgb, points3d = read_points3D_binary(bin_path)
        print("Converting point3d.bin to .ply.")
        storePly(ply_path, xyz, rgb)
    else:
        points3d = None

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, path=path)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)  
    if eval:
        if isinstance(train_list[0], str):
            print("read train/test image names from json file.\n")
            train_cam_infos = []
            test_cam_infos = []
            for cam in cam_infos:
                if cam.image_name in train_list:
                    train_cam_infos.append(cam)
                elif cam.image_name in test_list:
                    test_cam_infos.append(cam)
                else:
                    print(f"can not find image in the split file: {cam.image_name}\n")
        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_list] 
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_list]   
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           points3d = points3d,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
