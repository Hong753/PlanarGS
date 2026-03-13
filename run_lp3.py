import os
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from lp3.mask_refine import BoxSmaller, NormalSplit
from lp3.box_migrate import AddPreviosBox, FilterMask, LP3Cam
from scene.dataset_readers import readColmapSceneInfo
from lp3.run_groundedsam import GroundingDINO, SAM
from argparse import ArgumentParser
from arguments import ModelParams, PriorParams
from scene.cameras import LoadGeomprior
from common_utils.graphics_utils import get_k
from lp3.color_cluster import MaskDistance
from planar.visualize import visualSegmask, visualMask


def draw_boxes(image, boxes, labels):
    """
    Args:
        image: np.ndarray, HWC, BGR
        boxes: list of [x0, y0, x1, y1]
        labels: list of str
    Returns:
        image_with_boxes: np.ndarray
    """
    img = image.copy()
    
    for box, label in zip(boxes, labels):
        x0, y0 = int(box[0]), int(box[1])
        x1, y1 = int(box[2]), int(box[3])
        color = (0, 255, 0) if not label.startswith("[proj]") else (0, 0, 255)  # BGR
        thickness = 2
        cv2.rectangle(img, (x0, y0), (x1, y1), color=color, thickness=thickness)
        
        # draw texts
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        text_thickness = 2
        _, baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        cv2.putText(img, label, (x0, y0 + 2 * baseline), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return img


def LP3(model, prp, text_prompts, vis, device="cuda"):   # Set vis=True to visualize boxes & masks.
    input_folder = model.source_path
    output_folder = os.path.join(input_folder, "planarprior")    
    os.makedirs(output_folder, exist_ok=True)
    if prp.visdebug:
        debug_folder = os.path.join(output_folder, "debug")
        os.makedirs(debug_folder, exist_ok=True)

    detection = GroundingDINO(device)
    detection.load_model(prp.ckpt_det)

    segmentation = SAM(device)
    segmentation.load_model(prp.ckpt_seg)

    if os.path.exists(os.path.join(input_folder, "sparse")):
        scene_info = readColmapSceneInfo(input_folder, eval=False)
    else:
        assert False, "Could not recognize scene type!"
    
    cam_infos = scene_info.train_cameras
    add_previous = False
    
    for cam in tqdm(cam_infos, desc="Processing cameras"):
        file_path = os.path.join(cam.path, "images", cam.image_name[0] + cam.image_name[1])
        image_name = cam.image_name[0]
        geom_folder = os.path.join(cam.path, "geomprior")
        depth, normal = LoadGeomprior(geom_folder, image_name, cam.size)
        W, H = cam.size
        K, inv_K = get_k(cam.FovX, cam.FovY, H, W)
        if prp.visdebug:
            camdebug_folder = os.path.join(debug_folder, image_name)
            os.makedirs(camdebug_folder, exist_ok=True)
        else:
            camdebug_folder = None

        detection.load_image(file_path)
        boxes_filt, pred_phrases = detection.get_detection_output(text_prompts, with_logits=False)

        # The box is represented by its top-left and bottom-right vertices.
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        if add_previous:
            boxes_filt, pred_phrases = AddPreviosBox(pred_phrases, boxes_filt, cam, previous_cam, camdebug_folder)
        boxes_filt, pred_phrases = BoxSmaller(boxes_filt, pred_phrases)

        # Segmentation from boxes
        segmentation.load_image(file_path)
        masks = segmentation.get_segmentation_mask(boxes_filt)

        distance_mask = MaskDistance(depth, normal, inv_K, prp.dis_thresh)
        if prp.visdebug:
            visualMask(distance_mask, path=camdebug_folder, filename="distance")

        masks = masks * distance_mask
        masks, pred_phrases, boxes_filt = NormalSplit(masks, pred_phrases, boxes_filt, normal, prp.normal_split, camdebug_folder)
        
        masks_previous, pred_previous = FilterMask(masks, pred_phrases)
        add_previous = True
        previous_cam = LP3Cam(cam_info=cam, depth=depth, preds=pred_previous, masks=masks_previous)

        mask_folder = os.path.join(output_folder, "mask")
        os.makedirs(mask_folder, exist_ok=True)
        num = 1
        mask_use = torch.zeros(masks[0].shape).to(device)
        for mask in masks:
            mask_use = mask_use + mask * num
            num += 1
        np.save(os.path.join(mask_folder, image_name + ".npy"), mask_use.cpu().numpy())   

        if vis:
            image =  cv2.cvtColor(np.array(Image.open(file_path)), cv2.COLOR_RGB2BGR)
            vis_boxes_path = os.path.join(output_folder, "boxes_vis")
            vis_mask_path = os.path.join(output_folder, "mask_vis")
            os.makedirs(vis_boxes_path, exist_ok=True)
            os.makedirs(vis_mask_path, exist_ok=True)
            image_with_boxes = draw_boxes(image, boxes_filt, pred_phrases)
            cv2.imwrite(os.path.join(vis_boxes_path, image_name + ".png"), image_with_boxes)
            visualSegmask(mask_use.squeeze(0), path=vis_mask_path, filename=image_name)   # image can be added as background


if __name__ == "__main__":
    parser = ArgumentParser(description="LP3 script parameters")
    model = ModelParams(parser, sentinel=True)
    prp = PriorParams(parser)
    parser.add_argument('-t', '--text_prompts', type=str, default="wall. floor. door. screen. window. ceiling. table")
    parser.add_argument("--vis", action="store_true") 
    args = parser.parse_args()
                                                                                                                                                         
    LP3(model.extract(args), prp.extract(args), args.text_prompts, args.vis)
