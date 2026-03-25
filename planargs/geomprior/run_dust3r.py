import os
import torch
import numpy as np
import tempfile
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as pl
pl.ion()

import sys
sys.path.append(os.path.join(os.getcwd(), "submodules/dust3r"))

from submodules.dust3r.dust3r.inference import inference
from submodules.dust3r.dust3r.model import AsymmetricCroCo3DStereo
from submodules.dust3r.dust3r.image_pairs import make_pairs
from submodules.dust3r.dust3r.utils.image import load_images
from submodules.dust3r.dust3r.utils.device import to_numpy
from submodules.dust3r.dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from submodules.dust3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile, scene


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)


def get_reconstructed_scene(output_folder, vis, outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01
    niter = 300

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    outfile, trimesh_scene = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                                     clean_depth, transparent_cams, cam_size)
    
    # Display the scene using trimesh
    if vis:
        out_path = os.path.join(output_folder, 'scene.glb')
        trimesh_scene.export(file_obj = out_path)
        print(f"Model saved to {out_path}")

    # also return rgb, depth and confidence imgs
    depths = to_numpy(scene.get_depthmaps())

    os.makedirs(output_folder + "/depth/", exist_ok=True)
    for i in range(len(depths)):
        name, t = os.path.splitext(os.path.basename(filelist[i]))
        np.save(output_folder + "/depth/" + name + ".npy", depths[i])

    confs = to_numpy([c for c in scene.im_conf])
    os.makedirs(output_folder + "/confs/", exist_ok=True)
    for i in range(len(confs)):
        name, t = os.path.splitext(os.path.basename(filelist[i]))
        np.save(output_folder + "/confs/" + name + ".npy", confs[i])

    return scene, outfile, imgs


def main_demo(tmpdirname, model, device, image_size, input_folder, output_folder, image_list, vis, silent):
    recon_fun = functools.partial(get_reconstructed_scene, output_folder, vis, tmpdirname, model, device, silent, image_size)
    
    # List all files in the input folder
    if image_list is None:
        input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    else:
        input_files = [os.path.join(input_folder, f) for f in image_list if os.path.isfile(os.path.join(input_folder, f))]

    # Filter only image files (assuming common image extensions)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    input_files = [f for f in input_files if os.path.splitext(f)[1].lower() in image_extensions]

    # Check if any image files were found
    if not input_files:
        print("No image files found in the specified folder.")
        return

    # Reconstruction options (same as before)
    schedule = 'linear'
    niter = 'niter'
    scenegraph_type = 'complete'
    winsize = 1 
    refid = 1 
    min_conf_thr = 0.3
    cam_size = 0.05
    as_pointcloud = 1
    mask_sky = 0
    clean_depth = 0
    transparent_cams = 1

    scene, outfile, imgs = recon_fun(input_files, schedule, niter, min_conf_thr, as_pointcloud,
                                     mask_sky, clean_depth, transparent_cams, cam_size,
                                     scenegraph_type, winsize, refid)
    print(f"3D model saved to: {outfile}")


def DUSt3R(input_folder, output_folder, weights_path, image_list=None, vis=False, device='cuda', image_size=512, silent=False):
    # Hardcoded input folder
    tmp_path = './submodules/dust3r/tmp'
    os.makedirs(tmp_path, exist_ok=True)
    tempfile.tempdir = tmp_path

    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)

    with tempfile.TemporaryDirectory(suffix='dust3r_gradio_demo') as tmpdirname:
        if not silent:
            print('Outputing stuff in', tmpdirname)
        
        # Run the main demo with input folder specified
        main_demo(tmpdirname, model, device, image_size, input_folder, output_folder, image_list, vis, silent)
