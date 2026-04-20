#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import random
import sys
import os
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from common_utils.general_utils import safe_state
from common_utils.loss_utils import l1_loss, l2_loss, ssim
from planar.co_planar import co_planar
from arguments import ModelParams, PipelineParams, OptimizationParams, PriorParams
from planar.training_report import prepare_output_and_logger, training_report

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

def training(dataset, opt, pipe, prp, test_iters, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, prp)
    gaussians.training_setup(opt) 
    viewpoint_stack = scene.getTrainCameras().copy()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    # Planar-guided initialization
    elif prp.use_initdensify:  
        densify_num = prp.initdensify_num
        initdensify_idx = 0
        cam_num = len(viewpoint_stack)
        densify_image_list = []
        pbar = tqdm(total=densify_num, desc="Planar-guided initialization")
        # Densify only 50 perspectives to avoid out-of-memory issues
        if cam_num > densify_num:
            densify_num_list = random.sample(range(0, cam_num), densify_num)  
            num = 0
            for cam in viewpoint_stack:
                if num in densify_num_list:
                    densify_image_list.append(cam.image_name)
                num += 1
        else:
            for cam in viewpoint_stack:
                densify_image_list.append(cam.image_name)
            densify_num = cam_num

        for idx in range(cam_num):
            viewpoint_cam = viewpoint_stack[idx]
            if viewpoint_cam.image_name in densify_image_list:
                initdensify_idx += 1
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, return_plane=False, return_depth_normal=False) 
                vis_mask = (render_pkg["out_observe"] > 0) & render_pkg["visibility_filter"]
                gaussians.plane_initdensify(viewpoint_cam, vis_mask, prp)
                pbar.update(1)
                if (initdensify_idx == densify_num):
                    pbar.close()
                    print("\nFinish densifying gaussian in plane area.")
                    point_cloud_path = os.path.join(dataset.model_path, "point_cloud/planar_initdensify")
                    print(f"Saving Gaussians to {point_cloud_path}")
                    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
                    break

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_imageloss_for_log = 0.0
    ema_dnloss_for_log = 0.0
    ema_planar_for_log = 0.0   
    ema_priordep_for_log = 0.0
    ema_priornorm_for_log = 0.0

    plane_loss, prior_deploss, prior_normloss, dn_loss = None, None, None, None

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        #gt_image, prior_depth, prior_normal = viewpoint_cam.gt_image, #viewpoint_cam.priordepth, viewpoint_cam.priornormal
        #canny_mask, planarmasks, conf_mask = viewpoint_cam.canny_mask, #viewpoint_cam.planarmask, viewpoint_cam.depth_conf
        # Bei --data_device cpu liegen alle Kamera-Daten auf der CPU.
        # Fuer den Training-Loop muessen sie auf der GPU sein.
        gt_image = viewpoint_cam.gt_image.cuda()
        prior_depth = viewpoint_cam.priordepth.cuda() if viewpoint_cam.priordepth is not None else None
        prior_normal = viewpoint_cam.priornormal.cuda() if viewpoint_cam.priornormal is not None else None
        canny_mask = viewpoint_cam.canny_mask.cuda()
        planarmasks = viewpoint_cam.planarmask.cuda() if viewpoint_cam.planarmask is not None else None
        conf_mask = viewpoint_cam.depth_conf.cuda() if viewpoint_cam.depth_conf is not None else None
       
        # Iterations for depth/normal rendering
        iter_argu = min(opt.dnloss_iteration, opt.planar_iteration, opt.priordepth_iteration)
        iter_dn = min(opt.dnloss_iteration, opt.priornormal_iteration)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                            return_plane=(iteration >= iter_argu), return_depth_normal=(iteration >= iter_dn)) 
        image, visibility_filter, radii = render_pkg["render"], render_pkg["visibility_filter"], render_pkg["radii"]
        out_observe, viewspace_point_tensor_abs,  viewspace_point_tensor = render_pkg["out_observe"], render_pkg["viewspace_points_abs"], render_pkg["viewspace_points"]
        if iteration >= iter_argu:
            render_depth, render_normal = render_pkg["plane_depth"], render_pkg["rendered_normal"]
        if iteration >= iter_dn:
            depth_normal = render_pkg["depth_normal"]

        ### Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()

        # scale loss 
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.lambda_scale * min_scale_loss.mean()
        
        # depth prior loss
        if iteration >= opt.priordepth_iteration and prior_depth is not None:  
            depth_mask = (prior_depth > 0) 
            depth_prior_mask = canny_mask * conf_mask
            prior_deploss = l2_loss(prior_depth, render_depth * depth_mask, depth_prior_mask) * viewpoint_cam.depth_weight * opt.lambda_priordepth
            loss = loss + prior_deploss        

        # planar loss  
        if iteration >= opt.planar_iteration and planarmasks is not None:
            if planarmasks.max() != 0:
                planar_depth = co_planar(render_depth, planarmasks, viewpoint_cam.inv_K)
                planar_mask = (planarmasks > 0).int() 
                plane_loss = opt.lambda_planar * l1_loss(render_depth, planar_depth, planar_mask) 
                loss += plane_loss
                
            # prior normal loss 
            if iteration >= opt.priornormal_iteration and prior_normal is not None:
                ones = torch.ones((prior_normal.shape[1], prior_normal.shape[2])).cuda()
                prior_normloss_1 = l1_loss(prior_normal, depth_normal, planar_mask * depth_prior_mask)   
                prior_normloss_2 = torch.mean(((ones - torch.sum(prior_normal * depth_normal, dim=0)) * planar_mask * depth_prior_mask))
                prior_normloss = opt.lambda_priornormal * (prior_normloss_1 + prior_normloss_2)
                loss += prior_normloss
      
        # normal-consistency loss 
        if iteration > opt.dnloss_iteration:        
            if iteration >= opt.planar_iteration:
                dn_loss = opt.lambda_dnloss * l1_loss(render_normal, depth_normal, (1 - planar_mask) * canny_mask)
            else:
                dn_loss = opt.lambda_dnloss * l1_loss(render_normal, depth_normal, canny_mask)
            loss += dn_loss
        
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_imageloss_for_log = 0.4 * image_loss.item() + 0.6 * ema_imageloss_for_log
            ema_dnloss_for_log = 0.4 * dn_loss.item() + 0.6 * ema_dnloss_for_log if dn_loss is not None else 0.0 
            ema_planar_for_log = 0.4 * plane_loss.item() + 0.6 * ema_planar_for_log  if plane_loss is not None else 0.0 
            ema_priordep_for_log = 0.4 * prior_deploss.item() + 0.6 * ema_priordep_for_log if prior_deploss is not None else 0.0
            ema_priornorm_for_log = 0.4 * prior_normloss.item() + 0.6 * ema_priornorm_for_log if prior_normloss is not None else 0.0
            
            if iteration % 10 == 0:
                loss_dict = {
                    "ImageLoss": f"{ema_imageloss_for_log:.{5}f}",
                    "DNloss": f"{ema_dnloss_for_log:.{5}f}",
                    "Planeloss": f"{ema_planar_for_log:.{5}f}",
                    "PriorDeploss": f"{ema_priordep_for_log:.{5}f}",
                    "PriorNormloss": f"{ema_priornorm_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, dn_loss, plane_loss, prior_deploss, prior_normloss, co_planar, 
                            l1_loss, iter_start.elapsed_time(iter_end), test_iters, scene, render, (pipe, background), vis_planar=(iteration >= opt.planar_iteration))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # Densification
            if iteration < opt.densify_until_iter: 
                # Keep track of max radii in image-space for pruning
                mask = (out_observe > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:  
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)

            # reset_opacity
            if iteration < opt.densify_until_iter:  
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):  
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
    
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration % 500 == 0:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    pr = PriorParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 7000, 14000, 20000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)     
    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), pr.extract(args), args.test_iterations,  args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint)
    print("\nTraining complete.")


