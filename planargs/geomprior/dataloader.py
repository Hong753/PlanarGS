import os
import json
import numpy as np
from .regreader_utils import LoadGroupDepth

def clamp(x, min_v, max_v):
    return max(min_v, min(x, max_v))


def SaveDepthInfo(prep, all_depthinfo, geomprior_path):
    save_conffolder = os.path.join(geomprior_path, "resized_confs")
    save_depthfolder = os.path.join(geomprior_path, "aligned_depth")
    save_weights = os.path.join(geomprior_path, "depth_weights.json")
    save_normalfolder = os.path.join(geomprior_path, "prior_normal")
    os.makedirs(save_depthfolder, exist_ok=True)
    os.makedirs(save_normalfolder, exist_ok=True)
    os.makedirs(save_conffolder, exist_ok=True)

    if not os.path.exists(save_weights):
        with open(save_weights, 'w') as f:
            json.dump({}, f)
    with open(save_weights, 'r') as f:
        weights = json.load(f)

    loss_weights = []
    max_confs = []

    for idx in range(len(all_depthinfo)):
        depthinfo = all_depthinfo[idx]
        max_confs.append(depthinfo.depth_conf.max())
        loss_weights.append(clamp(depthinfo.depth_weight, prep.weights_min_thresh, prep.weights_max_thresh))

    max_conf = max(max_confs)
    norm_weights = 1 - (np.array(loss_weights) - prep.weights_min_thresh) / (prep.weights_max_thresh - prep.weights_min_thresh)

    for idx in range(len(all_depthinfo)):
        depthinfo = all_depthinfo[idx]
        cam_info = depthinfo.cam_info
        np.save(os.path.join(save_depthfolder, cam_info.image_name[0] + ".npy"), depthinfo.depth_aligned) 
        np.save(os.path.join(save_normalfolder, cam_info.image_name[0] + ".npy"), depthinfo.prior_normal) 
        depth_conf = depthinfo.depth_conf / max_conf
        np.save(os.path.join(save_conffolder, cam_info.image_name[0] + ".npy"), depth_conf)  

        norm_weight = norm_weights[idx]
        weights[cam_info.image_name[0]] = norm_weight
        with open(save_weights, 'w') as f:
            json.dump(weights, f, indent=4)


def GroupAlign(prep, cam_infos, points3d, geomprior_path, vis): 
    param_path = os.path.join(geomprior_path, "depth_param.json")
    if not os.path.exists(param_path):
        with open(param_path, 'w') as f:
            json.dump({}, f)
    with open(param_path, 'r') as f:
        params = json.load(f)

    group_folders = [
        name
        for name in os.listdir(geomprior_path)
        if name.startswith("_group")
        and os.path.isdir(os.path.join(geomprior_path, name))
    ]
    all_depthinfo = []

    for idx in range(len(group_folders)):
        group = group_folders[idx]
        group_path = os.path.join(geomprior_path, group) 
        groupdepth_folder = os.path.join(group_path, "depth") 
        groupconf_folder = os.path.join(group_path, "confs") 
        group_name = [
            os.path.splitext(f)[0]
            for f in os.listdir(groupdepth_folder)
            if os.path.isfile(os.path.join(groupdepth_folder, f))
            and f.endswith(".npy")
        ]
        group_cam_infos = [
            cam_info
            for cam_info in cam_infos
            if cam_info.image_name[0] in group_name
        ]
        
        print(f"\nStart aligning depth {group}:")
        if vis:
            vis_path = geomprior_path
        else:
            vis_path = None
        depthinfo_list, depth_params = LoadGroupDepth(group_cam_infos, groupdepth_folder, groupconf_folder, points3d, vis_path, prep)
        all_depthinfo.extend(depthinfo_list)

        params[group] = depth_params
        with open(param_path, 'w') as f:
            json.dump(params, f, indent=4)

    SaveDepthInfo(prep, all_depthinfo, geomprior_path)



        
            
