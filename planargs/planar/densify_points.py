import torch


def project_points_to_image(K, pointscam, width, height):
    cam_coord = torch.matmul(K, pointscam)

    projected_points = cam_coord / (cam_coord[2] + 1e-5)
    
    valid_mask = (cam_coord[2] > 0) & \
                 (projected_points[0] >= 0) & (projected_points[0] <= width - 1) & \
                 (projected_points[1] >= 0) & (projected_points[1] <= height - 1)
    
    valid_projected_points = projected_points[:2].T[valid_mask]
    
    return valid_projected_points, valid_mask


def PlaneMaskGS(points_3d, planarmasks, mask, K, R, T):
    result_mask = -torch.ones_like(mask, device="cuda", dtype=torch.int)
    selected_points = points_3d[mask]
    selected_indices = torch.nonzero(mask).squeeze(1) 
    h, w = planarmasks.shape
    pointscam = torch.matmul(R.T, selected_points.T) + T.reshape(3,1)

    valid_projected_points, valid_mask = project_points_to_image(K, pointscam, w, h)
    valid_selected_indices = selected_indices[valid_mask]

    u_coords = valid_projected_points[:, 0].long()
    v_coords = valid_projected_points[:, 1].long()
    labels = planarmasks[v_coords, u_coords]
    result_mask[valid_selected_indices] = labels.to(torch.int)  

    return result_mask

    
# Generate an intermediate M × N matrix, may cause GPU OOM when M and N are large.    
def find_nearest(object_points, all_points, mask=None):
    """
    object_points: M x 3 
    all_points:    N x 3 
    mask:          N 
    """
    if mask is not None:
        selected_points = all_points[mask]
        selected_indices = torch.nonzero(mask).squeeze(1)

        diff = object_points.unsqueeze(1) - selected_points.unsqueeze(0)  # M x N x 3
        dist2 = (diff ** 2).sum(-1)                                 # M x N
        nearest_selected = dist2.argmin(dim=1)
        nearest_indices = selected_indices[nearest_selected]
    else:
        # M x N
        diff = object_points.unsqueeze(1) - all_points.unsqueeze(0)  # M x N x 3
        dist2 = (diff ** 2).sum(-1)                                 # M x N
        nearest_indices = dist2.argmin(dim=1)

    return nearest_indices


def InitialPlaneSeg(planarmasks):
    seg_mask = planarmasks.view(1, -1)
    seg_num = seg_mask.max().item() + 1  
     # The number of pixels for each semantic class.
    seg_pnum = torch.zeros((1, seg_num), dtype=torch.int64).to("cuda")  
    seg_pnum.scatter_add_(1, seg_mask, torch.ones_like(seg_mask))

    return seg_mask, seg_num, seg_pnum


def SegPoints(i, seg_pnum, seg_mask, points3d):
    temp_mask = torch.nonzero(seg_mask == i)[...,1].view(1, -1) 
    points_seg = torch.zeros((3, seg_pnum[0, i])).to("cuda") 

    for j in range(3):
        points_seg[j,...] = torch.take(points3d[j,...], temp_mask)

    return points_seg, temp_mask