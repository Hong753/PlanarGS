import torch
from planargs.common_utils.graphics_utils import Depth2Pointscam
from planargs.planar.densify_points import InitialPlaneSeg, SegPoints


def co_planar(depth, segmask, inv_K):
    '''
    Re-generate the planar-refined depth map.
    '''      
    # Ensure gradients are computable by removing edge regions.
    segmask[:8, :] = 0
    segmask[-8:,:] = 0
    segmask[:, :8] = 0
    segmask[:,-8:] = 0
        
    points3D, coords = Depth2Pointscam(depth, inv_K)
    plane_depth = ProjectDepth(points3D, segmask, inv_K, depth, coords)  
         
    return plane_depth

        

def ProjectDepth(points3d, segmask, inv_K, depth, coords):
    '''
    input: points3d (k, 3) ; depth (H, W) 

    Step 1: Estimate plane parameters from the point cloud using least squares.
            plane_param = (points3D x points3D^T + delt * E)_inv x points3D x Ym
            3x1            3xk        kx3          3x3             3xk        kx1

    Step 2: Reconstruct the depth map from the estimated plane parameters.
            depth = 1 / (plane_params^T x inv_K x coords)
            1xk          1x3              3x3     3xk

    return: plane_depth hxw   adjust_mask hxw
    '''
    seg_mask, seg_num, seg_pnum = InitialPlaneSeg(segmask)
    plane_depth = torch.zeros((1, seg_mask.shape[1])).cuda()  
    adjust_mask = torch.zeros((1, seg_mask.shape[1]), dtype = torch.int).cuda()
    
    for i in range(1, seg_num):
        points_seg, temp_mask = SegPoints(i, seg_pnum, seg_mask, points3d.T)  # 3xk

        # Step 1
        # Perform least-squares for each planar region.
        delt = 0.01
        pTp = torch.matmul(points_seg, points_seg.T) + delt * torch.eye(3).cuda() 
        pTp_inv = torch.inverse(pTp)
        pTp_inv_pT = torch.matmul(pTp_inv, points_seg) 
        plane_param = torch.matmul(pTp_inv_pT, torch.ones((points_seg.shape[1], 1)).cuda()) 
        
        # Step 2
        coords_seg = torch.zeros((3,seg_pnum[0, i])).cuda() 
        for j in range(3):
            coords_seg[j,...] = torch.take(coords[j,...], temp_mask)  

        invk_coords = torch.matmul(inv_K, coords_seg)
        pdepth_seg = 1.0 / (torch.matmul(plane_param.T, invk_coords))  

        plane_depth.scatter_(1, temp_mask, pdepth_seg)

    return plane_depth.view(depth.shape[0], depth.shape[1])


