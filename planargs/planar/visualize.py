import numpy as np
import cv2 as cv
from PIL import Image
import os
import torch


def AlphaImg(bg, over, alpha=0.6):
    """
    Overlay the transparent overlay image onto the background image.
    """
    alpha_channel = np.full((over.shape[0], over.shape[1]), 255, dtype=np.uint8)
    overlay = cv.merge((over, alpha_channel))

    overlay_rgb = overlay[:, :, :3]
    overlay_alpha = overlay[:, :, 3] / 255.0  
    overlay_alpha *= alpha
    result = (overlay_rgb * overlay_alpha[:, :, np.newaxis] + bg * (1 - overlay_alpha[:, :, np.newaxis])).astype(np.uint8)
    return result


def visualDepth(depths, path=None, filename=None, miner=None, maxer=None):
    depth = depths.clone().detach().cpu().numpy()

    if miner is None and maxer is None:
        mask = (depth == 0).astype(int)
        if depth.min() == 0:
            miner = (depth + depth.max() * mask).min()
        else:
            miner = depth.min()
        maxer = depth.max()

    depth_normalized = (depth - miner) / (maxer - miner) * 255
    im_depth = cv.applyColorMap(
        cv.convertScaleAbs(depth_normalized, alpha=1),
        cv.COLORMAP_JET
    )
    im_masked = (depth != 0).astype(np.uint8)[:, :, np.newaxis].repeat(3, axis=2) * im_depth

    if path is not None and filename is not None:
        cv.imwrite(os.path.join(path, filename + ".jpg"), im_masked)

    return miner, maxer, torch.from_numpy(im_masked).permute(2, 0, 1)


def visualDepthGray(depths, path, filename):
    depth = depths.clone().detach().cpu().numpy()
    mask = (depth == 0).astype(int)
    if depth.min() == 0:
        miner = (depth + depth.max() * mask).min()
    else:
        miner = depth.min()
    depth_normalized = (depth - miner) * (mask == 0).astype(int) / (depth.max() - miner) * 255    
    im_depth = Image.fromarray((depth_normalized.astype(int)) .astype(np.uint8))
    im_depth.save(os.path.join(path, filename + ".jpg"))
    
    
def visualMask(mask, path=None, filename=None):
    masker = (mask * 255).to(torch.uint8)

    if path is not None and filename is not None:
        masker_pic = Image.fromarray(masker.clone().detach().cpu().numpy())
        masker_pic.save(os.path.join(path, filename + ".jpg"))

    return masker.unsqueeze(0).repeat(3, 1, 1)


def visualNorm(normal, path=None, filename=None):
    normal_pic = (((normal + 1.0) * 0.5).clamp(0, 1) * 255).to(torch.uint8)

    if path is not None and filename is not None:
        normal_image = Image.fromarray(normal_pic.permute(1, 2, 0).clone().detach().cpu().numpy())
        normal_image.save(os.path.join(path, filename + ".jpg"))

    return normal_pic


def visualSegmask(segmasker, path=None, filename=None, img=None):
    '''
    img: background image  np.ndarray
    '''
    num = torch.max(segmasker).item()   
    H, W = segmasker.shape
    bg = torch.zeros((H, W, 3), dtype=torch.uint8, device=segmasker.device)

    while num > 0:
        show_mask = (segmasker == num).to(torch.uint8)  # H x W
        num -= 1

        color = torch.randint(
            0, 256, (3,), dtype=torch.uint8, device=segmasker.device
        )  # (3,)

        mask_rgb = show_mask.unsqueeze(-1).repeat(1, 1, 3)  # H x W x 3
        colored_mask = mask_rgb * color.view(1, 1, 3)
        bg = bg + colored_mask

    bg_save = bg.clone().detach().cpu().numpy()

    if img is not None:
        bg_save = AlphaImg(img, bg_save)
   
    if path is not None and filename is not None:
        fig_path = os.path.abspath(os.path.join(path, filename + ".jpg"))
        cv.imwrite(fig_path, bg_save)

    return bg.permute(2, 0, 1)

