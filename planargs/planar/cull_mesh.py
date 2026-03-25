import numpy as np
import torch
from planargs.scene.dataset_readers import readColmapSceneInfo
from planargs.common_utils.graphics_utils import get_k


def cull_mesh(rec_mesh, data_path, trans=None, eval=False):
    scene_info = readColmapSceneInfo(data_path, eval)
    cam_infos = scene_info.train_cameras
    points = torch.from_numpy(rec_mesh.vertices).float().cuda()

    # # delete mesh vertices that are not inside any training camera's viewing frustum
    whole_mask = np.ones(points.shape[0]).astype(bool)
    for cam in cam_infos: 
        R = torch.from_numpy(cam.R).float().cuda().T
        T = torch.from_numpy(cam.T).float().cuda().reshape(3,1)
        W, H = cam.size
        K, inv_K = get_k(cam.FovX, cam.FovY, H, W)

        if "scale" in trans.keys():
            points *= trans["scale"]

        if trans is not None:
            R = R @ torch.inverse(trans["Rd"])
            T = -R @ trans["Td"] + T

        cam_coord = torch.matmul(K, (torch.matmul(R, points.T) + T))
    
        # NDC space
        projected_points = cam_coord / (cam_coord[2] + 1e-5)
        
        valid_mask = (cam_coord[2] > 0) & \
                    (projected_points[0] >= 0) & (projected_points[0] <= W - 1) & \
                    (projected_points[1] >= 0) & (projected_points[1] <= H - 1)

        whole_mask &= ~valid_mask.cpu().numpy()  # whole_mask indicates the points that need to be removed
    
    face_mask = whole_mask[rec_mesh.faces].all(axis=1)
    rec_mesh.update_faces(~face_mask)

    return rec_mesh
    
   
def mask_mesh(mesh_gt, mesh_rec, recon_path, threshold):
    min_points = mesh_gt.vertices.min(axis=0)
    max_points = mesh_gt.vertices.max(axis=0)
    min_bbox = min_points * threshold
    max_bbox = max_points * threshold

    ### Regarding the incomplete ground-truth mesh in the MuSHRoom dataset.
    if 'classroom' in recon_path:
        min_bbox[1] = min_points[1] * 0.95 
    if 'kokko' in recon_path:
        min_bbox[2] = min_points[2] * 0.98 
        max_bbox[2] = max_points[2] * 1.03 
    if 'vr_room' in recon_path:
        min_bbox[1] = min_points[1] *0.95 

    mask_min = (mesh_rec.vertices - min_bbox[None]) > 0
    mask_max = (mesh_rec.vertices - max_bbox[None]) < 0
    mask = np.concatenate((mask_min, mask_max), axis=1).all(axis=1)
    face_mask = mask[mesh_rec.faces].all(axis=1)
    mesh_rec.update_vertices(mask)
    mesh_rec.update_faces(face_mask)
    
    return mesh_rec
