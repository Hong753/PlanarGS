import numpy as np
import torch
from planargs.common_utils.graphics_utils import Pointscam2Depth, Depth2Pointscam, get_k
from dataclasses import dataclass
from planargs.scene.colmap_loader import Camera
from planargs.planar.visualize import visualSegmask

@dataclass
class LP3Cam:
    cam_info: Camera
    depth: torch.tensor
    preds: list
    masks: list


### Keep only the three largest masks by area.
def FilterMask(masker, pred_phrases, n=3):
    masks = []
    largest_masks = []
    pred_1 = []
    for i in range(masker.shape[0]):
        masks.append(masker[i].squeeze(0).int())

    areas = [(mask.sum(), idx) for idx, mask in enumerate(masks)]
    areas.sort(key=lambda x: x[0], reverse=True)
    largest_indices = [idx for _, idx in areas[:n]]
    for idx in range(len(masks)):
        if idx in largest_indices:
            proj_pred = pred_phrases[idx].replace("[proj]", "", 1)
            largest_masks.append(masks[idx]) 
            pred_1.append(f"[proj]{proj_pred}")
    
    return largest_masks, pred_1


def AddPreviosBox(pred_phrases, boxes_filt, cam_info, previous_cam, debug_folder=None):
    """
    boxes_filt: torch.Tensor, shape: [num_boxes, 4]  
    pred_phrases: List[str], shape: [num_boxes]  'prompt'
    masker: list - np.numpy(height, width)
    """
    boxes_filter = []
    for i in range(boxes_filt.shape[0]):
        boxes_filter.append(boxes_filt[i])
    new_boxes = []
    new_pred = []

    masks_previous = previous_cam.masks
    if debug_folder is not None:
        vis_proj = torch.zeros(cam_info.size[1], cam_info.size[0]).cuda()

    for num in range(len(masks_previous)):
        mask = masks_previous[num]
        mask_proj = ProjectPlane(mask, previous_cam.depth, previous_cam.cam_info, cam_info)
        ys, xs = torch.where(mask_proj)
        if len(xs) == 0:
            continue  
        x1 = xs.min()
        y1 = ys.min()
        x2 = xs.max()
        y2 = ys.max()
        new_boxes.append(torch.stack([x1, y1, x2, y2]).int().to(boxes_filt.device))
        new_pred.append(previous_cam.preds[num])

        if debug_folder is not None:
            vis_proj = vis_proj + (mask_proj * (num + 1))

    boxes_list = boxes_filter + new_boxes
    pred_phrases = pred_phrases + new_pred
    if debug_folder is not None:
        visualSegmask(vis_proj, path=debug_folder, filename="proj_previous_mask")

    return torch.stack(boxes_list), pred_phrases


def ProjectPlane(mask_src, depth_src, cam_info_src, cam_info_tgt):
    # from depth & mask to points
    K_src, inv_K_src = get_k(cam_info_src.FovX, cam_info_src.FovY, cam_info_src.size[1], cam_info_src.size[0])
    K_tgt, inv_K_tgt = get_k(cam_info_tgt.FovX, cam_info_tgt.FovY, cam_info_tgt.size[1], cam_info_tgt.size[0])
    points_src, _ = Depth2Pointscam(depth_src, inv_K_src, mask_src)
    device = depth_src.device
    H, W = depth_src.shape
    
    # transfer points from src to tgt
    R_src = torch.inverse(torch.from_numpy(cam_info_src.R).float().to(device))
    t_src = -R_src @ torch.from_numpy(cam_info_src.T).float().to(device).view(3, 1)
    R_tgt = torch.inverse(torch.from_numpy(cam_info_tgt.R).float().to(device))
    t_tgt = -R_tgt @ torch.from_numpy(cam_info_tgt.T).float().to(device).view(3, 1)
    R_rel = R_tgt @ torch.inverse(R_src)
    t_rel = t_tgt - R_tgt @ R_src.T @ t_src
    points_3d_tgt = points_src @ R_rel.T + t_rel.T  # (N, 3)

    # project points to mask
    mask_tgt, _  = Pointscam2Depth(K_tgt, points_3d_tgt.T, (H, W))

    return mask_tgt


