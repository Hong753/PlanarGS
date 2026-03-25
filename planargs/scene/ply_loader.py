
import numpy as np
from planargs.scene.gaussian_model import BasicPointCloud
from typing import NamedTuple
from dataclasses import dataclass
from planargs.common_utils.graphics_utils import getWorld2View2
import numpy as np
from plyfile import PlyData, PlyElement

@dataclass
class CameraInfo:
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    path: str
    image_name: str
    size: list
    points3d_ids: np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    points3d: np.array
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0  if 'red' in vertices else None
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T  if 'nx' in vertices else None
    # error_init = np.zeros((normals.shape[0]))
    errors = vertices['xyzerr']/(np.min(vertices['xyzerr'] + 1e-8))  if 'xyzerr' in vertices else None
    pcd = BasicPointCloud(points=positions, colors=colors, normals=normals, errors=errors)
    return pcd


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
