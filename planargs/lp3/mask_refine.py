import re
import torch
from planargs.lp3.color_cluster import SplitPic, MaxConnect
from planargs.planar.visualize import visualSegmask


def equal(a, b):
    return abs(a - b) < 3

def is_contained(box1, box2):
    """
    Return True if box1 is fully inside box2, otherwise False.
    """
    return (((box1[0] > box2[0]) or equal(box1[0], box2[0])) and ((box1[1] > box2[1]) or equal(box1[1], box2[1])) and 
    (box1[2] < box2[2] or equal(box1[2], box2[2])) and (box1[3] <= box2[3] or equal(box1[3], box2[3])))


### For boxes with the same name that are nested, keep only the smaller one.
def BoxSmaller(boxes_filt, pred_phrases):
    """
    boxes_filt: torch.Tensor, shape: [num_boxes, 4]  
    pred_phrases: List[str], shape: [num_boxes]  
    """
    boxes = boxes_filt.to(torch.int)
    boxes_list = [box.clone() for box in boxes]
    box_names = pred_phrases.copy()
    for i in range(len(box_names)):
        num = len(box_names)
        if i >= num:
            break
        if box_names[i].replace("[proj]", "", 1) == "wall":
            continue
        for j in range(i+1, len(box_names)):
            m = len(box_names)
            if j >= m:
                break
            if box_names[i].replace("[proj]", "", 1) == box_names[j].replace("[proj]", "", 1):
                if is_contained(boxes[i], boxes[j]):
                    boxes_list.pop(j)
                    box_names.pop(j)
                elif is_contained(boxes[j], boxes[i]):
                    boxes_list.pop(i)
                    box_names.pop(i)

    return torch.stack(boxes_list), box_names


def NormalSplit(masks, pred_phrases, boxes_filt, normal, cluster, camdebug_folder=None):
    """
    masks: torch.Tensor, shape: [num_masks, 1, height, width]
    """
    if masks.shape[0] != len(pred_phrases):
        print("error! Masks num not equal to phrases num.")

    normal = normal / (torch.linalg.norm(normal, dim=-1, keepdim=True) + 1e-8)
    normal_rgb = (normal + 1.0) * 127.5
    color_masks = SplitPic(normal_rgb, cluster)  
    new_masks = []
    new_pred = []
    boxes = []

    if camdebug_folder is not None:
        vis_colormask = torch.zeros(normal.shape[0], normal.shape[1], dtype=torch.uint8, device=normal.device)
        idx = 1
        for color_mask in color_masks:
            vis_colormask += color_mask * idx
            idx += 1
        visualSegmask(vis_colormask, path=camdebug_folder, filename=f"normal_split")
        
    for i in range(masks.shape[0]):
        mask = masks[i, 0]
        mask1, mask2 = MaxOverlap(mask, color_masks)
        new_mask = mask1 * mask
        new_masks.append(new_mask)
        new_pred.append(pred_phrases[i])
        boxes.append(boxes_filt[i])
        if mask2 is not None:
            new_mask = mask2 * mask
            new_masks.append(new_mask)
            new_pred.append(pred_phrases[i])
            boxes.append(boxes_filt[i])

    new_masks, new_pred, boxes = adjust_masks(new_masks, new_pred, boxes)
    new_masks = torch.stack(new_masks).unsqueeze(1).bool()

    return new_masks, new_pred, boxes


def OverlapnContain(mask_a, mask_b, threshold = 0.05):
    """
    Compute the overlap ratio between two masks and determine whether one fully contains the other.
    Returns:
        float: The proportion of mask A that overlaps with mask B.
        float: The proportion of mask B that overlaps with mask A.
        bool: Whether mask A completely contains mask B.
        bool: Whether mask B completely contains mask A.
    """
    mask_a_bool = mask_a.bool()
    mask_b_bool = mask_b.bool()

    intersection = torch.logical_and(mask_a_bool, mask_b_bool)

    area_a = mask_a_bool.sum()
    area_b = mask_b_bool.sum()
    overlap_area = intersection.sum()

    ratio_a = overlap_area / area_a if area_a > 0 else torch.tensor(0.0, device=mask_a.device)
    ratio_b = overlap_area / area_b if area_b > 0 else torch.tensor(0.0, device=mask_b.device)

    containment_a_in_b = ratio_a > (1 - threshold)
    containment_b_in_a = ratio_b > (1 - threshold)
    
    return ratio_a, ratio_b, containment_a_in_b, containment_b_in_a


def adjust_masks(masks, pred, boxes, thresh = 2000):
    new_mask = []
    new_pred = []
    new_box = []
    
    i = 0
    while i < len(masks):
        j = i + 1
        i_exists = True
        while (j < len(masks)) & i_exists:
            ratio_i, ratio_j, contain_i_in_j, contain_j_in_i = OverlapnContain(masks[i], masks[j])
            
            if contain_i_in_j:
                masks.pop(i)
                pred.pop(i)
                boxes.pop(i)
                i_exists = False
            elif contain_j_in_i: 
                masks.pop(j)
                pred.pop(j)
                boxes.pop(j)
            else:
                if ratio_i > ratio_j and ratio_i > 0:
                    masks[j] = torch.logical_and(masks[j], torch.logical_not(masks[i]))
                elif ratio_j > 0:
                    masks[i] = torch.logical_and(masks[i], torch.logical_not(masks[j]))
                j = j + 1

        if i_exists & (masks[i].sum() < thresh):
            masks.pop(i)
            pred.pop(i)
            boxes.pop(i)
            i_exists = False

        if i_exists:    
            max_mask, second_mask = MaxConnect(masks[i])
            if max_mask is None:
                masks.pop(i)
                pred.pop(i)
                boxes.pop(i)
                i_exists = False
            else:
                masks[i] = max_mask
                if second_mask is not None:
                    new_mask.append(second_mask)
                    new_pred.append(pred[i])
                    new_box.append(boxes[i])
                i = i + 1
    return masks + new_mask, pred + new_pred, boxes + new_box



def MaxOverlap(mask, mask_list, area_threshold=0.7):
    areas = []
    mask_bool = mask.bool()
    for idx, m in enumerate(mask_list):
        m_bool = m.bool()
        area = torch.logical_and(mask_bool, m_bool).sum()
        areas.append((area, idx))

    areas.sort(key=lambda x: x[0], reverse=True)
    if len(areas) < 2 or areas[0][0] == 0:
        return mask_list[areas[0][1]], None

    max_area, max_idx = areas[0]
    second_max_area, second_max_idx = areas[1]
    ratio = (max_area - second_max_area) / max_area

    if ratio > area_threshold:
        return mask_list[max_idx], None
    else:
        return mask_list[max_idx], mask_list[second_max_idx]
